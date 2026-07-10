from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import statistics
import threading
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
except Exception:
    SymSpell = None
    Verbosity = None

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None

try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore
except Exception:
    AnalyzerEngine = None

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None

from .config import settings
from .seed import DEMO_MEETINGS


WORD_TOKEN_PATTERN = r"[a-zA-Z][a-zA-Z0-9-]+"


def word_tokens(text: str) -> list[str]:
    return re.findall(WORD_TOKEN_PATTERN, text)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_to_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value)


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(text: str, chunk_words: int, overlap: int) -> list[str]:
    """Fixed-window fallback chunker."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _carry_overlap(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing sentences that fit within the overlap word budget."""
    carry: list[str] = []
    carry_words = 0
    for s in reversed(sentences):
        w = len(s.split())
        if carry_words + w <= overlap:
            carry.insert(0, s)
            carry_words += w
        else:
            break
    return carry, carry_words


def _encode_sentences(sentences: list[str], embedder: Any) -> Any:
    """Encode sentences; load default model if embedder is None."""
    if embedder is None:
        if SentenceTransformer is None:
            return None
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def semantic_chunk_text(
    text: str,
    chunk_words: int,
    overlap: int,
    similarity_threshold: float = 0.75,
    embedder: Any = None,
) -> list[str]:
    """Semantic chunking using sentence embeddings; falls back to fixed-window if unavailable."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]
    if len(sentences) < 3:
        return chunk_text(text, chunk_words, overlap)

    try:
        embeddings = _encode_sentences(sentences, embedder)
        if embeddings is None:
            return chunk_text(text, chunk_words, overlap)
    except Exception:
        return chunk_text(text, chunk_words, overlap)

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        word_count = len(sentences[i].split())
        should_split = (
            (float(np.dot(embeddings[i - 1], embeddings[i])) < similarity_threshold
             or (current_words + word_count) > chunk_words)
            and current_words >= 30
        )
        if should_split:
            chunks.append(". ".join(current) + ".")
            carry, carry_words = _carry_overlap(current, overlap)
            current = carry + [sentences[i]]
            current_words = carry_words + word_count
        else:
            current.append(sentences[i])
            current_words += word_count

    if current:
        chunks.append(". ".join(current) + ".")

    return [c for c in chunks if c.strip()]


def top_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = [token.lower() for token in word_tokens(text)]
    # Comprehensive stop-word list — prevents noise words showing as topics
    stopwords = {
        # articles / prepositions / conjunctions
        "the", "a", "an", "and", "or", "but", "nor", "so", "yet", "for",
        "to", "of", "in", "on", "at", "by", "as", "is", "it", "be",
        "into", "onto", "from", "with", "about", "against", "between",
        "through", "during", "before", "after", "above", "below",
        "than", "then", "that", "this", "these", "those", "such",
        # pronouns
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
        "his", "she", "her", "they", "them", "their", "it", "its",
        "who", "whom", "whose", "which", "what", "where", "when", "how",
        # common verbs / auxiliaries
        "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "must", "can", "could",
        "are", "was", "were", "been", "being", "get", "got", "go",
        "said", "say", "says", "also", "just", "need", "needs",
        # common filler / meeting procedural / conversational
        "there", "here", "very", "quite", "rather", "now", "well",
        "yes", "no", "not", "one", "two", "three", "any", "all",
        "some", "other", "each", "every", "own", "same", "so",
        "into", "onto", "up", "out", "if", "then", "because",
        "thank", "thanks", "welcome", "please", "okay", "right",
        "meeting", "agenda", "item", "items", "page", "number",
        "members", "member", "council", "committee",
        # spoken/conversational filler keywords
        "like", "know", "yeah", "think", "kind", "something", "want", "going", 
        "really", "people", "good", "much", "time", "little", "things", "thing", 
        "mean", "guess", "actually", "basically", "probably", "maybe", "sort", 
        "stuff", "sure", "definitely", "everyone", "someone", "anyone", 
        "everything", "anything", "nothing", "talk", "talking", "discuss", 
        "discussing", "see", "look", "make", "take", "give", "back", "first", 
        "last", "next", "start", "end", "use", "using", "work", "working", 
        "needed", "feel", "feeling", "find", "finding", "getting", "way", 
        "lot", "lots", "put", "putting", "come", "coming", "call", "called", 
        "tell", "told", "try", "trying", "tried", "done", "went", "gone", 
        "keep", "keeping", "kept", "show", "showing", "shown", "came",
        "even", "still", "always", "never", "another", "many", "more", 
        "less", "most", "least", "different", "seems", "seem", "seemed", 
        "made", "took", "gave", "given", "found", "let", "allow", "allows", 
        "used", "user", "users", "works", "new", "old", "part", "parts", 
        "point", "points", "problem", "problems", "question", "questions", 
        "answer", "answers", "tells", "ask", "asks", "asked", "thinks", 
        "thought", "thoughts", "knows", "knew", "known", "understand", 
        "understanding", "understood", "believe", "believes", "believed", 
        "sees", "saw", "seen", "looks", "wants", "wanted", "feels", "felt", 
        "tries", "uses", "makes", "making", "takes", "taking", "gets", 
        "goes", "comes", "gives", "giving", "finds", "keeps", "starts", 
        "starting", "started", "ends", "ending", "ended", "shows", 
        "calls", "run", "runs", "running", "ran", "hold", "holds", 
        "holding", "held", "write", "writes", "writing", "wrote", 
        "written", "read", "reads", "reading", "study", "studies", 
        "studying", "studied", "learn", "learns", "learning", "learnt", 
        "learned", "teach", "teaches", "teaching", "taught", "instruct", 
        "instructs", "instructing", "instructed", "train", "trains", 
        "training", "trained", "practice", "practices", "practicing", 
        "practiced", "exercise", "exercises", "exercising", "exercised",
        "rest", "rests", "resting", "rested", "sleep", "sleeps", 
        "sleeping", "slept", "wake", "wakes", "waking", "woke", "woken",
        "eat", "eats", "eating", "ate", "eaten", "drink", "drinks", 
        "drinking", "drank", "drunk", "cook", "cooks", "cooking", 
        "cooked", "bake", "bakes", "baking", "baked", "feed", "feeds", 
        "feeding", "fed", "wash", "washs", "washing", "washed", 
        "clean", "cleans", "cleaning", "cleaned", "dirty", "dirties", 
        "dirtying", "dirtied", "open", "opens", "opening", "opened", 
        "close", "closes", "closing", "closed", "lock", "locks", 
        "locking", "locked", "unlock", "unlocks", "unlocking", 
        "unlocked", "shut", "shuts", "shutting", "push", "pushs", 
        "pushing", "pushed", "pull", "pulls", "pulling", "pulled", 
        "throw", "throws", "throwing", "threw", "thrown", "catch", 
        "catches", "catching", "caught", "carry", "carries", "carrying", 
        "carried", "drop", "drops", "dropping", "dropped", "fall", 
        "falls", "falling", "fell", "fallen", "rise", "rises", "rising", 
        "rose", "risen", "jump", "jumps", "jumping", "jumped", "leap", 
        "leaps", "leaping", "leaped", "climb", "climbs", "climbing", 
        "climbed", "descend", "descends", "descending", "descended", 
        "slide", "slides", "sliding", "slid", "slip", "slips", 
        "slipping", "slipped", "roll", "rolls", "rolling", "rolled", 
        "spin", "spins", "spinning", "spun", "turn", "turns", "turning", 
        "turned", "bend", "bends", "bending", "bent", "shake", "shakes", 
        "shaking", "shook", "shaken", "tremble", "trembles", 
        "trembling", "trembled", "shiver", "shivers", "shivering", 
        "shivered", "freeze", "freezes", "freezing", "froze", 
        "frozen", "melt", "melts", "melting", "melted", "burn", 
        "burns", "burning", "burned", "burnt", "heat", "heats", 
        "heating", "heated", "cool", "cools", "cooling", "cooled", 
        "warm", "warms", "warming", "warmed", "chill", "chills", 
        "chilling", "chilled", "wet", "wets", "wetting", "wetted", 
        "dry", "dries", "drying", "dried", "fill", "fills", "filling", 
        "filled", "empty", "empties", "emptying", "emptied", "pour", 
        "pours", "pouring", "poured", "flow", "flows", "flowing", 
        "flowed", "leak", "leaks", "leaking", "leaked", "drip", 
        "drips", "dripping", "dripped", "spill", "spills", "spilling", 
        "spilt", "spilled", "mix", "mixes", "mixing", "mixed", 
        "blend", "blends", "blending", "blended", "stir", "stirs", 
        "stirring", "stirred", "combine", "combines", "combining", 
        "combined", "separate", "separates", "separating", "separated", 
        "divide", "divides", "dividing", "divided", "split", "splits", 
        "splitting", "cut", "cuts", "cutting", "chop", "chops", 
        "chopping", "chopped", "slice", "slices", "slicing", "sliced", 
        "carve", "carves", "carving", "carved", "tear", "tears", 
        "tearing", "tore", "torn", "rip", "rips", "ripping", "ripped", 
        "break", "breaks", "breaking", "broke", "broken", "smash", 
        "smashes", "smashing", "smashed", "crush", "crushes", 
        "crushing", "crushed", "grind", "grinds", "grinding", "ground", 
        "powder", "powders", "powdering", "powdered", "solidify", 
        "solidifies", "solidifying", "solidified", "vaporize", 
        "vaporizes", "vaporizing", "vaporized", "condense", 
        "condenses", "condensing", "condensed", "evaporate", 
        "evaporates", "evaporating", "evaporated", "dissolve", 
        "dissolves", "dissolving", "dissolved", "absorb", "absorbs", 
        "absorbing", "absorbed", "emit", "emits", "emitting", 
        "emitted", "reflect", 'reflects', 'reflecting', 'reflected', 
        'transmit', 'transmits', 'transmitting', 'transmitted',
        'conduct', 'conducts', 'conducting', 'conducted', 'insulate', 
        'insulates', 'insulating', 'insulated', 'magnetize', 
        'magnetizes', 'magnetizing', 'magnetized', 'electrify',
        'electrifies', 'electrifying', 'electrified', 'charge', 
        'charges', 'charging', 'charged', 'discharge', 'discharges', 
        'discharging', 'discharged', 'shock', 'shocks', 'shocking',
        'shocked', 'strike', 'strikes', 'striking', 'struck', 'hit', 
        'hits', 'hitting', 'beat', 'beats', 'beating', 'beaten', 
        'whip', 'whips', 'whipping', 'whipped', 'punch', 'punches', 
        'punching', 'punched', 'slap', 'slaps', 'slapping', 'slapped',
        'kick', 'kicks', 'kicking', 'kicked', 'bite', 'bites', 
        'biting', 'bitten', 'scratch', 'scratches', 'scratching', 
        'scratched', 'pinch', 'pinches', 'pinching', 'pinched',
        'squeeze', 'squeezes', 'squeezing', 'squeeze', 'squeezed', 
        'hug', 'hugs', 'hugging', 'hugged', 'kiss', 'kisses', 
        'kissing', 'kissed', 'touch', 'touches', 'touching', 
        'touched', 'grasp', 'grasps', 'grasping', 'grasped', 
        'clutch', 'clutches', 'clutching', 'clutched', 'grip', 
        'grips', 'gripping', 'gripped', 'bind', 'binds', 'binding', 
        'bound', 'tie', 'ties', 'tying', 'tied', 'untie', 'unties', 
        'untying', 'untied', 'loose', 'looses', 'loosing', 'loosed', 
        'unloose', 'unlooses', 'unloosing', 'unloosed', 'fasten', 
        'fastens', 'fastening', 'fastened', 'unfasten', 'unfastens', 
        'unfastening', 'unfastened', 'attach', 'attachs', 'attaching', 
        'attached', 'detach', 'detachs', 'detaching', 'detached', 
        'disconnect', 'disconnects', 'disconnecting', 'disconnected', 
        'link', 'links', 'linking', 'linked', 'unlink', 'unlinks', 
        'unlinking', 'unlinked', 'couple', 'couples', 'coupling', 
        'coupled', 'uncouple', 'uncouples', 'uncoupling', 'uncoupled', 
        'disjoin', 'disjoins', 'disjoining', 'disjoined', 'unite', 
        'unites', 'uniting', 'united', 'disunite', 'disunites', 
        'disuniting', 'disunited', 'merge', 'merges', 'merging', 
        'merged', 'fuse', 'fuses', 'fusing', 'fused', 'divorce', 
        'divorces', 'divorcing', 'divorced', 'marry', 'marries', 
        'marrying', 'married', 'wed', 'weds', 'wedding', 'wedded', 
        'betroth', 'betroths', 'betrothing', 'betrothed', 'promise', 
        'promises', 'promising', 'promised', 'swear', 'swears', 
        'swearing', 'swore', 'sworn', 'vow', 'vows', 'vowing', 
        'vowed', 'pledge', 'pledges', 'pledging', 'pledged', 
        'guarantee', 'guarantees', 'guaranteeing', 'guaranteed', 
        'warrant', 'warrants', 'warranting', 'warranted', 'ensure', 
        'ensures', 'ensuring', 'ensured', 'secure', 'secures', 
        'securing', 'secured', 'defend', 'defends', 'defending', 
        'defended', 'guard', 'guards', 'guarding', 'guarded', 
        'shield', 'shields', 'shielding', 'shielded', 'shelter', 
        'shelters', 'sheltering', 'sheltered', 'harbor', 'harbors', 
        'harboring', 'harbored', 'haven', 'havens', 'refuge', 
        'refuges', 'sanctuary', 'sanctuaries', 'asylum', 'asylums', 
        'safety', 'safeties', 'security', 'securities', 'protection', 
        'protections', 'defense', 'defenses', 'barrier', 'barriers', 
        'wall', 'walls', 'fence', 'fences', 'gate', 'gates', 'door', 
        'doors', 'window', 'windows', 'roof', 'roofs', 'ceiling', 
        'ceilings', 'floor', 'floors', 'room', 'rooms', 'house', 
        'houses', 'home', 'homes', 'building', 'buildings', 
        'structure', 'structures', 'construction', 'constructions', 
        'architecture', 'architectures', 'blueprint', 'blueprints', 
        'map', 'maps', 'chart', 'charts', 'graph', 'graphs', 
        'diagram', 'diagrams', 'illustration', 'illustrations', 
        'picture', 'pictures', 'photo', 'photos', 'photograph', 
        'photographs', 'sketch', 'sketches', 'portrait', 
        'portraits', 'landscape', 'landscapes', 'sculpture', 
        'sculptures', 'statue', 'statues', 'monument', 'monuments', 
        'memorial', 'memorials', 'tomb', 'tombs', 'grave', 'graves', 
        'cemetery', 'cemeteries', 'church', 'churches', 'temple', 
        'temples', 'mosque', 'mosques', 'synagogue', 'synagogues', 
        'shrine', 'shrines', 'altar', 'altars', 'pulpit', 'pulpits', 
        'pew', 'pews', 'font', 'fonts', 'aisle', 'aisles', 'nave', 
        'naves', 'transept', 'transepts', 'choir', 'choirs', 
        'chancel', 'chancels', 'crypt', 'crypts', 'vault', 'vaults', 
        'dome', 'domes', 'spire', 'spires', 'steeple', 'steeples', 
        'tower', 'towers', 'turret', 'turrets', 'minaret', 
        'minarets', 'pagoda', 'pagodas', 'obelisk', 'obelisks', 
        'pyramid', 'pyramids', 'sphinx', 'sphinxes', 'colossus', 
        'colossuses', 'landmark', 'landmarks', 'beacon', 'beacons', 
        'lighthouse', 'lighthouses', 'mast', 'masts', 'pole', 
        'poles', 'post', 'posts', 'pillar', 'pillars', 'column', 
        'columns', 'arch', 'arches', 'bridge', 'bridges', 'viaduct', 
        'viaducts', 'aqueduct', 'aqueducts', 'tunnel', 'tunnels', 
        'channel', 'channels', 'canal', 'canals', 'ditch', 'ditches', 
        'trench', 'trenches', 'moat', 'moats', 'dyke', 'dykes', 
        'dam', 'dams', 'reservoir', 'reservoirs', 'lake', 'lakes', 
        'pond', 'ponds', 'pool', 'pools', 'basin', 'basins', 'well', 
        'wells', 'fountain', 'fountains', 'spring', 'springs', 
        'geyser', 'geysers', 'hot_spring', 'hot_springs', 'spa', 
        'spas', 'bath', 'baths', 'shower', 'showers', 'sink', 
        'sinks', 'tap', 'taps', 'faucet', 'faucets', 'pipe', 
        'pipes', 'tube', 'tubes', 'hose', 'hoses', 'valve', 
        'valves', 'pump', 'pumps', 'drain', 'drains', 'sewer', 
        'sewers', 'gutter', 'gutters', 'spout', 'spouts', 'funnel', 
        'funnels', 'filter', 'filters', 'strainer', 'strainers', 
        'sieve', 'sieves', 'grate', 'grates', 'grill', 'grills', 
        'grid', 'grids', 'lattice', 'lattices', 'screen', 'screens', 
        'mesh', 'meshes', 'net', 'nets', 'network', 'networks', 
        'web', 'webs', 'cobweb', 'cobwebs', 'spider_web', 
        'spider_webs', 'loom', 'looms', 'weave', 'weaves', 'weaving', 
        'wove', 'woven', 'knit', 'knits', 'knitting', 'knitted', 
        'stitch', 'stitchs', 'stitching', 'stitched', 'embroider', 
        'embroiders', 'embroidering', 'embroidered', 'quilt', 
        'quilts', 'quilting', 'quilted', 'patch', 'patchs', 
        'patching', 'patched', 'darn', 'darns', 'darning', 'darned', 
        'renew', 'renews', 'renewing', 'renewed', 'restore', 
        'restores', 'restoring', 'restored'
    }
    # Require at least 4 characters AND not a stopword
    counts = Counter(
        token for token in tokens
        if token not in stopwords and len(token) >= 4
    )
    return [term for term, _ in counts.most_common(limit)]


from html.parser import HTMLParser

class _DDGHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current_result: dict[str, str] | None = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "") or ""
        cls_split = cls.split()

        if tag == "div" and "result" in cls_split:
            if "result--ad" not in cls_split:
                self.current_result = {"title": "", "url": "", "snippet": ""}

        if self.current_result:
            if tag == "a" and "result__a" in cls_split:
                self.in_title = True
                href = attrs_dict.get("href", "") or ""
                if "uddg=" in href:
                    parsed = urllib.parse.urlparse(href)
                    qs = urllib.parse.parse_qs(parsed.query)
                    real_url = qs.get("uddg", [href])[0]
                    self.current_result["url"] = real_url
                else:
                    if href.startswith("//"):
                        href = "https:" + href
                    self.current_result["url"] = href

            elif tag == "a" and "result__snippet" in cls_split:
                self.in_snippet = True

    def handle_data(self, data: str) -> None:
        if self.current_result:
            if self.in_title:
                self.current_result["title"] += data
            elif self.in_snippet:
                self.current_result["snippet"] += data

    def handle_endtag(self, tag: str) -> None:
        if self.current_result:
            if tag == "a" and self.in_title:
                self.in_title = False
            elif tag == "a" and self.in_snippet:
                self.in_snippet = False
                self.current_result["title"] = re.sub(r"\s+", " ", self.current_result["title"]).strip()
                self.current_result["snippet"] = re.sub(r"\s+", " ", self.current_result["snippet"]).strip()
                if self.current_result["title"] and self.current_result["url"] and self.current_result["snippet"]:
                    self.results.append(self.current_result)
                self.current_result = None


@dataclass
class RetrievedChunk:
    meeting_id: str
    meeting_title: str
    chunk_id: str
    chunk_index: int
    text: str
    sanitized_text: str
    score: float
    keywords: list[str]


class AuragKnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Use thread-local storage for SQLite connections so each thread gets
        # its own connection to the same database file (avoids InterfaceError).
        self._local = threading.local()
        self._whisper_model = None
        self._llm = None
        self._embedder = None
        self._presidio = None
        self._symspell = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._ensure_schema()
        self.seed_demo_content()
        self.meetings = self._load_meetings()
        self._migrate_meetings_if_needed()
        # Build lightweight BM25 index synchronously so the API is immediately usable.
        self._rebuild_bm25_only()
        # Load embedding models and build dense index in background.
        threading.Thread(target=self._prewarm_models, daemon=True).start()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection, creating one if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(settings.db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _rebuild_bm25_only(self) -> None:
        """Fast synchronous init: build chunk rows and BM25 only (no model downloads)."""
        self._refresh_meeting_chunks_in_memory()
        self._refresh_chunk_rows()
        self.chunk_texts = [row["sanitized_text"] for row in self.chunk_rows]
        self._bm25 = BM25Okapi([text.lower().split() for text in self.chunk_texts]) if BM25Okapi and self.chunk_texts else None
        self._vectorizer = None
        self._dense_matrix = None
        self._faiss_index = None
        self._embedder = None

    def _prewarm_models(self) -> None:
        """Load Whisper and embedding models in a background thread at startup."""
        try:
            embedder = self._get_embedder()
            if embedder is not None and self.chunk_texts:
                embeddings = embedder.encode(
                    self.chunk_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=32,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
                with self._lock:
                    self._embedder = embedder
                    self._dense_matrix = embeddings
                    if faiss is not None:
                        index = faiss.IndexFlatIP(embeddings.shape[1])
                        try:
                            resources = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(resources, 0, index)
                        except Exception:
                            pass
                        index.add(embeddings)
                        self._faiss_index = index
        except Exception:
            pass
        try:
            self._get_whisper()
        except Exception:
            pass
        try:
            self._get_llm()
        except Exception:
            pass

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists meetings (
                id text primary key,
                title text not null,
                audio_name text,
                created_at text not null,
                transcript text not null,
                sanitized_transcript text not null,
                summary text not null,
                action_items_json text not null,
                topics_json text not null,
                timeline_json text not null,
                redactions_json text not null,
                questions_json text not null,
                chunks_json text not null,
                model_trace_json text not null,
                wer_estimate real not null,
                snr_estimate real,
                pii_count integer not null,
                processing_status text not null,
                quality_flag text not null
            );
            create table if not exists users (
                id text primary key,
                email text unique not null,
                name text not null,
                password_hash text not null,
                created_at text not null
            );
            """
        )
        self.conn.commit()
        for migration in [
            "alter table meetings add column lecture_notes_json text not null default '[]'",
            "alter table meetings add column speaker_transcript text not null default ''",
            "alter table meetings add column user_id text not null default ''",
        ]:
            try:
                self.conn.execute(migration)
                self.conn.commit()
            except Exception:
                pass

    def _clean_repetitions(self, text: str) -> str:
        """Collapses consecutive repetitions of words or phrases (n-grams from 1 to 8 words)
        that repeat consecutively more than twice."""
        words = text.split()
        if not words:
            return text
        
        n = len(words)
        i = 0
        cleaned_words = []
        
        while i < n:
            matched_repeat = False
            for length in range(1, 9):
                if i + length * 2 <= n:
                    phrase = words[i:i+length]
                    repeat_count = 1
                    while i + length * (repeat_count + 1) <= n:
                        next_phrase = words[i + length * repeat_count : i + length * (repeat_count + 1)]
                        phrase_clean = [w.lower().strip(".,;:?!()[]{}'\"") for w in phrase]
                        next_phrase_clean = [w.lower().strip(".,;:?!()[]{}'\"") for w in next_phrase]
                        if phrase_clean == next_phrase_clean:
                            repeat_count += 1
                        else:
                            break
                    
                    if repeat_count > 1:
                        cleaned_words.extend(phrase)
                        i += length * repeat_count
                        matched_repeat = True
                        break
            
            if not matched_repeat:
                cleaned_words.append(words[i])
                i += 1
                
        return " ".join(cleaned_words)

    def _migrate_meetings_if_needed(self) -> None:
        """Clean filler word topics, repetition loops, and refresh database columns for existing meetings if needed."""
        updated = False
        filler_words = {"like", "know", "yeah", "think", "kind", "something"}
        with self._lock:
            for meeting in self.meetings:
                topics = meeting.get("topics", [])
                has_fillers = any(t.lower() in filler_words for t in topics)
                transcript = meeting.get("transcript", "")
                
                # Check if the transcript has repetition loops (e.g. clean_repetitions changes it significantly)
                cleaned_tx = self._clean_repetitions(transcript) if transcript else ""
                has_repetitions = len(cleaned_tx) < len(transcript) - 20 if (transcript and cleaned_tx) else False
                
                if (has_fillers or not topics or has_repetitions) and transcript:
                    meeting["transcript"] = cleaned_tx
                    meeting["sanitized_transcript"] = self._clean_repetitions(meeting.get("sanitized_transcript", ""))
                    
                    # Regenerate topics using the updated top_keywords function on the cleaned transcript
                    new_topics = self._topics(cleaned_tx)
                    meeting["topics"] = new_topics
                    
                    # Update chunks as well
                    chunks = meeting.get("chunks", [])
                    new_chunks = []
                    for idx, chunk in enumerate(chunks):
                        txt = chunk.get("text", "")
                        cleaned_chunk = self._clean_repetitions(txt)
                        new_chunks.append({
                            "chunk_id": chunk.get("chunk_id", f"{meeting['id']}-chunk-{idx}"),
                            "chunk_index": idx,
                            "text": cleaned_chunk,
                            "sanitized_text": cleaned_chunk,
                            "keywords": top_keywords(cleaned_chunk, limit=5),
                        })
                    meeting["chunks"] = new_chunks
                    
                    # Update timeline and lecture notes
                    meeting["timeline"] = self._build_timeline(cleaned_tx)
                    meeting["lecture_notes"] = self._build_lecture_notes(new_chunks)
                    
                    self._write_meeting_to_db(meeting)
                    updated = True
            if updated:
                rows = self.conn.execute("select * from meetings order by created_at desc").fetchall()
                meetings = []
                for row in rows:
                    m = dict(row)
                    m["created_at"] = iso_to_dt(m["created_at"])
                    m["action_items"] = json_load(m.pop("action_items_json"), [])
                    m["topics"] = json_load(m.pop("topics_json"), [])
                    m["timeline"] = json_load(m.pop("timeline_json"), [])
                    m["redactions"] = json_load(m.pop("redactions_json"), [])
                    m["questions"] = json_load(m.pop("questions_json"), [])
                    m["chunks"] = json_load(m.pop("chunks_json"), [])
                    m["model_trace"] = json_load(m.pop("model_trace_json"), {})
                    meetings.append(m)
                self.meetings = meetings

    def seed_demo_content(self) -> None:
        if self.conn.execute("select count(*) from meetings").fetchone()[0]:
            return
        for meeting in DEMO_MEETINGS:
            record = {
                **meeting,
                "sanitized_transcript": meeting["transcript"],
                "redactions": [],
                "questions": [],
                "chunks": [],
                "model_trace": {"pipeline": "seed"},
                "wer_estimate": 0.08,
                "snr_estimate": 28.0,
                "pii_count": 0,
                "processing_status": "ready",
                "quality_flag": "green",
            }
            self.save_meeting(record)

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if SentenceTransformer is None:
            return None
        try:
            self._embedder = SentenceTransformer(settings.embed_model)
        except Exception:
            self._embedder = None
        return self._embedder

    def _get_presidio(self):
        if self._presidio is not None:
            return self._presidio
        if AnalyzerEngine is None:
            return None
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"project_name": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
            self._presidio = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception:
            try:
                self._presidio = AnalyzerEngine()
            except Exception:
                self._presidio = None
        return self._presidio

    def _get_symspell(self):
        if self._symspell is not None:
            return self._symspell
        if SymSpell is None:
            return None
        try:
            symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            for term in {
                "meeting", "audio", "transcript", "quality", "index", "search", "answer",
                "question", "redaction", "redacted", "dashboard", "summary", "action",
                "items", "speaker", "confidence", "retrieval", "hybrid", "symspell",
                "bm25", "faiss", "upload", "export", "pdf", "history",
            }:
                symspell.create_dictionary_entry(term, 1000)
            self._symspell = symspell
        except Exception:
            self._symspell = None
        return self._symspell

    def _get_whisper(self):
        if self._whisper_model is not None:
            return self._whisper_model
        if WhisperModel is None:
            return None
        try:
            import os
            self._whisper_model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                cpu_threads=os.cpu_count() or 4,
                num_workers=2,
            )
        except Exception:
            self._whisper_model = None
        return self._whisper_model

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if Llama is None or not settings.llm_model_path:
            return None
        model_path = Path(settings.llm_model_path)
        if not model_path.exists():
            return None
        try:
            n_threads = 1 if settings.llm_n_gpu_layers != 0 else (os.cpu_count() or 4)
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.llm_n_ctx,
                n_threads=n_threads,
                n_batch=512,
                logits_all=False,
                verbose=False,
                n_gpu_layers=settings.llm_n_gpu_layers,
            )
        except Exception:
            self._llm = None
        return self._llm

    def _regex_findings(self, text: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        patterns = [
            (r"[\w.+-]+@[\w-]+\.[\w.-]+", "EMAIL_ADDRESS", 0.99, "regex"),
            (r"\+?\d[\d\s().-]{7,}\d", "PHONE_NUMBER", 0.94, "regex"),
        ]
        for pattern, entity_type, confidence, method in patterns:
            for match in re.finditer(pattern, text):
                findings.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "entity_type": entity_type,
                        "text": match.group(0),
                        "replacement": f"[{entity_type}]",
                        "confidence": confidence,
                        "method": method,
                    }
                )
        return findings

    # Entity types Presidio should redact. Only genuinely personal/confidential categories —
    # PERSON is excluded so speaker names stay visible; DATE_TIME/LOCATION/NRP/URL are excluded
    # because they're generic (a time of day, a city, a nationality) rather than identifying.
    _PRESIDIO_SKIP_ENTITIES = {"PERSON", "DATE_TIME", "LOCATION", "NRP", "URL"}

    # Below this score, pattern-only recognizers (e.g. US_DRIVER_LICENSE matching a bare
    # short alphanumeric token like "S3") are context-less guesses, not confident PII hits.
    _PRESIDIO_MIN_CONFIDENCE = 0.5

    def _presidio_findings(self, text: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        presidio = self._get_presidio()
        if presidio is None:
            return findings
        try:
            results = presidio.analyze(text=text, language="en")
            for result in results:
                if result.entity_type in self._PRESIDIO_SKIP_ENTITIES:
                    continue
                if float(getattr(result, "score", 0.0) or 0.0) < self._PRESIDIO_MIN_CONFIDENCE:
                    continue
                findings.append(
                    {
                        "start": result.start,
                        "end": result.end,
                        "entity_type": result.entity_type,
                        "text": text[result.start : result.end],
                        "replacement": f"[{result.entity_type}]",
                        "confidence": float(getattr(result, "score", 0.75) or 0.75),
                        "method": "presidio",
                    }
                )
        except Exception:
            pass
        return findings

    def _dedupe_findings(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[tuple[int, int, str], dict[str, Any]] = {}
        for finding in findings:
            key = (finding["start"], finding["end"], finding["entity_type"])
            if key not in unique:
                unique[key] = finding
        return sorted(unique.values(), key=lambda item: item["start"])

    def _apply_findings(self, text: str, findings: list[dict[str, Any]]) -> str:
        redacted = text
        for finding in sorted(findings, key=lambda item: item["start"], reverse=True):
            redacted = redacted[: finding["start"]] + finding["replacement"] + redacted[finding["end"] :]
        return redacted

    def _redaction_log(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(findings),
            "by_type": dict(Counter(item["entity_type"] for item in findings)),
            "methods": sorted({item["method"] for item in findings}),
        }

    def _corrections(self, text: str) -> tuple[str, bool]:
        symspell = self._get_symspell()
        if symspell is None:
            return text, False
        corrected_tokens: list[str] = []
        changed = False
        for token in text.split():
            clean = re.sub(r"[^a-zA-Z]", "", token)
            if len(clean) < 3:
                corrected_tokens.append(token)
                continue
            suggestions = symspell.lookup(clean.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].distance > 0:
                corrected = suggestions[0].term
                corrected_tokens.append(corrected.capitalize() if token[:1].isupper() else corrected)
                changed = True
            else:
                corrected_tokens.append(token)
        return " ".join(corrected_tokens), changed

    def _redact(self, text: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        findings = self._regex_findings(text) + self._presidio_findings(text)
        merged = self._dedupe_findings(findings)
        redacted = self._apply_findings(text, merged)
        return redacted, merged, self._redaction_log(merged)

    def _estimate_snr(self, file_path: Path) -> float | None:
        if sf is None:
            return None
        try:
            samples, sample_rate = sf.read(str(file_path), always_2d=False)
            if samples.size == 0:
                return None
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            samples = np.asarray(samples, dtype=np.float32)
            frame_size = max(int(sample_rate * 0.03), 1)
            frame_count = len(samples) // frame_size
            if frame_count == 0:
                return None
            frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
            rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-9
            signal = np.percentile(rms, 80)
            noise = max(np.percentile(rms, 20), 1e-9)
            return float(20.0 * np.log10(signal / noise))
        except Exception:
            return None

    def _estimate_wer(
        self,
        snr: float | None,
        transcript: str,
        avg_logprob: float = -1.0,
        no_speech_prob: float = 0.0,
    ) -> float:
        # Whisper logprob signal: maps -0.0 → 0.0 WER, -0.5 → ~0.13, -1.0 → ~0.30, -2.0 → ~0.65
        # avg_logprob of -1.0 is used as sentinel for "no real signal" (fallback path)
        logprob_is_real = avg_logprob > -0.95  # real Whisper logprob, not the fallback -1.0
        if logprob_is_real:
            # Sigmoid-based mapping: excellent (>-0.2) → ~0.03, ok (-0.5) → ~0.15, bad (<-1.0) → ~0.5
            logprob_wer = float(max(0.03, min(0.90, 1.0 / (1.0 + math.exp((avg_logprob + 0.4) / 0.25)))))
            # Penalise if a large fraction of segments had no-speech (hallucination risk)
            speech_penalty = max(0.0, no_speech_prob - 0.3) * 0.5
            logprob_wer = min(0.95, logprob_wer + speech_penalty)

        if snr is None:
            if logprob_is_real:
                return logprob_wer
            return 0.10
        length_factor = min(len(transcript.split()) / 200.0, 1.0)
        quality_factor = 1.0 / (1.0 + math.exp((snr - 18.0) / 4.0))
        snr_wer = float(max(0.03, min(0.95, 0.05 + 0.65 * quality_factor + 0.08 * length_factor)))

        if logprob_is_real:
            # Blend SNR-based and logprob-based estimates; weight logprob more (it's directly from the model)
            return float(max(0.03, min(0.95, 0.35 * snr_wer + 0.65 * logprob_wer)))
        return snr_wer

    def _strip_silence(self, file_path: Path) -> tuple[Path, bool, float | None]:
        """
        Remove silent frames so Whisper only processes speech.
        Returns (path_to_use, was_temp_file, snr_estimate).
        Computes SNR from the already-loaded audio to avoid a second disk read.
        Caller must delete the temp file when done.
        Falls back to original path on any error.
        """
        if sf is None:
            return file_path, False, None
        try:
            import tempfile
            samples, sr = sf.read(str(file_path), always_2d=False)
            if samples.size == 0:
                return file_path, False, None
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            samples = np.asarray(samples, dtype=np.float32)

            # ── SNR estimate (from already-loaded samples — no second disk read) ────
            snr: float | None = None
            try:
                frame_size = max(int(sr * 0.03), 1)
                frame_count = len(samples) // frame_size
                if frame_count > 0:
                    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
                    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
                    signal = np.percentile(rms, 80)
                    noise = max(np.percentile(rms, 20), 1e-9)
                    snr = float(20.0 * np.log10(signal / noise))
            except Exception:
                pass

            # 20 ms frames; threshold = 1% of peak amplitude
            frame_size = max(int(sr * 0.02), 1)
            peak = np.max(np.abs(samples))
            if peak < 1e-6:
                return file_path, False, snr
            threshold = 0.01 * peak

            # Build a boolean mask marking frames that contain speech
            n_frames = len(samples) // frame_size
            speech_mask = np.zeros(len(samples), dtype=bool)
            pad = int(sr * 0.15)  # 150 ms padding around each speech frame
            for i in range(n_frames):
                s = i * frame_size
                e = s + frame_size
                if np.sqrt(np.mean(samples[s:e] ** 2)) > threshold:
                    lo = max(0, s - pad)
                    hi = min(len(samples), e + pad)
                    speech_mask[lo:hi] = True

            speech_ratio = float(speech_mask.sum()) / max(len(speech_mask), 1)
            # Not worth stripping if >80% is already speech
            if speech_ratio > 0.80:
                return file_path, False, snr

            speech_samples = samples[speech_mask]
            if speech_samples.size < sr:  # < 1 s of speech — skip
                return file_path, False, snr

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sf.write(tmp.name, speech_samples, sr)
            return Path(tmp.name), True, snr
        except Exception:
            return file_path, False, None

    # MLX model IDs (Apple Silicon GPU).
    # Covers all standard Whisper sizes. Any model NOT in this map falls through to faster-whisper.
    _MLX_MODEL_MAP = {
        "tiny":      "mlx-community/whisper-tiny-mlx",
        "tiny.en":   "mlx-community/whisper-tiny.en-mlx",
        "base":      "mlx-community/whisper-base-mlx",
        "base.en":   "mlx-community/whisper-base.en-mlx",
        "small":     "mlx-community/whisper-small-mlx",
        "small.en":  "mlx-community/whisper-small.en-mlx",
        "medium":    "mlx-community/whisper-medium-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "large":     "mlx-community/whisper-large-v3-mlx",
        "large-v2":  "mlx-community/whisper-large-v2-mlx",
        "large-v3":  "mlx-community/whisper-large-v3-mlx",
    }

    def _transcribe(self, file_path: Path, progress_callback=None, cancel_event: threading.Event | None = None) -> dict[str, Any]:
        # Strip silence before transcription — reduces audio length by 20-50% for real meetings.
        # SNR is computed for free here from the already-loaded audio samples.
        audio_path, is_temp, precomputed_snr = self._strip_silence(file_path)
        try:
            result = self._transcribe_inner(audio_path, file_path, progress_callback, cancel_event)
            # Attach precomputed SNR so the caller doesn't need to re-read the file.
            if precomputed_snr is not None:
                result["precomputed_snr"] = precomputed_snr
            return result
        finally:
            if is_temp:
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    def _transcribe_inner(self, audio_path: Path, orig_path: Path, progress_callback=None, cancel_event: threading.Event | None = None) -> dict[str, Any]:
        # ── Try mlx-whisper (Apple Silicon GPU) first ─────────────────────────
        try:
            import mlx_whisper  # type: ignore
            mlx_model = self._MLX_MODEL_MAP.get(settings.whisper_model)
            if mlx_model is None:
                raise ImportError(f"No MLX model mapping for '{settings.whisper_model}', using faster-whisper")
            raw = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=mlx_model,
                temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False,
                verbose=False,
            )
            segments: list[dict] = []
            for seg in raw.get("segments", []):
                if cancel_event and cancel_event.is_set():
                    break
                text_cleaned = self._clean_repetitions(seg.get("text", "").strip())
                segments.append({
                    "start": float(seg.get("start", 0.0)),
                    "end":   float(seg.get("end", 0.0)),
                    "text":  text_cleaned,
                    "avg_logprob":   float(seg.get("avg_logprob", 0.0)),
                    "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
                })
                if progress_callback:
                    progress_callback(len(segments), float(seg.get("end", 0.0)))
            transcript = self._clean_repetitions(raw.get("text", "").strip())
            if not transcript:
                transcript = f"No speech detected in {orig_path.name}."
            return {
                "transcript": transcript,
                "segments": segments,
                "language": raw.get("language", "en"),
                "avg_logprob": float(statistics.mean(s["avg_logprob"] for s in segments)) if segments else -1.0,
                "no_speech_prob": float(statistics.mean(s["no_speech_prob"] for s in segments)) if segments else 0.0,
                "transcription_engine": "mlx-whisper",
            }
        except ImportError:
            pass  # mlx-whisper not installed — fall through to faster-whisper
        except Exception:
            pass  # mlx failed for another reason — fall through

        # ── Fall back to faster-whisper (CPU) ─────────────────────────────────
        whisper = self._get_whisper()
        if whisper is None:
            fallback = (
                f"Processed meeting from {orig_path.name}. Whisper is not configured, so this is a demo transcript. "
                "The app still indexes the content, applies redaction, and serves hybrid Q&A."
            )
            return {
                "transcript": fallback,
                "segments": [],
                "language": "en",
                "avg_logprob": -1.0,
                "no_speech_prob": 0.0,
                "transcription_engine": "fallback",
            }
        try:
            def _collect(path: Path, vad: bool) -> tuple[list[str], list[dict], list[float], list[float], Any]:
                segs, inf = whisper.transcribe(
                    str(path),
                    vad_filter=vad,
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    word_timestamps=False,
                    language="en",
                )
                texts, meta, logprobs, no_sp = [], [], [], []
                for seg in segs:
                    if cancel_event and cancel_event.is_set():
                        break
                    text_cleaned = self._clean_repetitions(seg.text.strip())
                    texts.append(text_cleaned)
                    meta.append({
                        "start": float(getattr(seg, "start", 0.0)),
                        "end":   float(getattr(seg, "end", 0.0)),
                        "text":  text_cleaned,
                        "avg_logprob":   float(getattr(seg, "avg_logprob", 0.0)),
                        "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0)),
                    })
                    logprobs.append(float(getattr(seg, "avg_logprob", 0.0)))
                    no_sp.append(float(getattr(seg, "no_speech_prob", 0.0)))
                    if progress_callback:
                        progress_callback(len(texts), getattr(seg, "end", 0.0))
                return texts, meta, logprobs, no_sp, inf

            text_segments, collected_segments, avg_logprob, no_speech, info = _collect(audio_path, vad=True)
            if not text_segments:
                text_segments, collected_segments, avg_logprob, no_speech, info = _collect(audio_path, vad=False)

            transcript = " ".join(part for part in text_segments if part).strip()
            transcript = self._clean_repetitions(transcript)
            if not transcript:
                transcript = f"No speech detected in {orig_path.name}. The audio may be silent or in an unsupported format."

            return {
                "transcript": transcript,
                "segments": collected_segments,
                "language": getattr(info, "language", "en"),
                "avg_logprob": float(statistics.mean(avg_logprob)) if avg_logprob else -1.0,
                "no_speech_prob": float(statistics.mean(no_speech)) if no_speech else 0.0,
                "transcription_engine": "whisper",
            }
        except Exception:
            fallback = f"Processed meeting from {orig_path.name}. The whisper backend failed, so the content is kept as a fallback transcript."
            return {
                "transcript": fallback,
                "segments": [],
                "language": "en",
                "avg_logprob": -1.0,
                "no_speech_prob": 0.0,
                "transcription_engine": "fallback",
            }

    # ── Speaker diarization ────────────────────────────────────────────────────

    def _diarize(self, file_path: Path, segments: list[dict]) -> list[dict]:
        """
        Assign speaker labels to transcript segments via resemblyzer embeddings + k-means.
        Returns a new list of segments with a 'speaker' field.
        Silently returns the original segments (without speaker) on any error.
        """
        if not segments or len(segments) < 2:
            return segments
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
            from sklearn.cluster import KMeans
            import numpy as np

            wav = preprocess_wav(str(file_path))
            sr = 16_000  # resemblyzer always resamples to 16 kHz

            encoder = VoiceEncoder()
            embeddings: list = []
            valid_idx: list[int] = []

            for i, seg in enumerate(segments):
                start = int(seg["start"] * sr)
                end = int(seg["end"] * sr)
                if end - start < sr // 4:  # skip clips shorter than 0.25 s
                    continue
                chunk = wav[start:end]
                if len(chunk) == 0:
                    continue
                embeddings.append(encoder.embed_utterance(chunk))
                valid_idx.append(i)

            if len(embeddings) < 2:
                return segments

            arr = np.array(embeddings)

            # Pick k: try 2–4, choose by largest relative inertia drop (elbow)
            max_k = min(4, len(embeddings))
            if max_k < 2:
                return segments

            best_k = 2
            if len(embeddings) >= 6 and max_k > 2:
                inertias = []
                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, random_state=42, n_init=3)
                    km.fit(arr)
                    inertias.append(km.inertia_)
                drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
                best_k = drops.index(max(drops)) + 2

            labels = KMeans(n_clusters=best_k, random_state=42, n_init=3).fit_predict(arr)
            speaker_map = {cluster: f"Speaker {chr(65 + cluster)}" for cluster in range(best_k)}

            result = [dict(seg) for seg in segments]
            idx_to_label = {valid_idx[i]: labels[i] for i in range(len(valid_idx))}
            for i, seg in enumerate(result):
                seg["speaker"] = speaker_map.get(idx_to_label.get(i, 0), "Speaker A")

            return result
        except Exception:
            return segments

    @staticmethod
    def _build_speaker_transcript(segments: list[dict]) -> str:
        """
        Produce a readable transcript with speaker labels, grouping consecutive segments
        from the same speaker into a single paragraph.
        """
        if not segments or not segments[0].get("speaker"):
            return ""
        lines: list[str] = []
        current_speaker = ""
        current_parts: list[str] = []
        for seg in segments:
            speaker = seg.get("speaker", "Speaker A")
            text = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                if current_parts:
                    lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
                current_speaker = speaker
                current_parts = [text]
            else:
                current_parts.append(text)
        if current_parts:
            lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
        return "\n\n".join(lines)

    # Phrases that mark procedural/formality noise (skip these sentences in summary)
    _PROCEDURAL_PHRASES = [
        "welcome to", "thank you", "this meeting", "members of", "good morning",
        "good afternoon", "good evening", "my name is", "before we begin",
        "we are now", "being recorded", "closed circuit", "monitored and",
        "on the agenda", "agenda is", "page of your", "apologies for absence",
        "roll call", "call to order", "meeting is called", "in the chair",
        "members present", "attendance",
    ]

    def _clean_disfluencies(self, text: str) -> str:
        fillers = [
            (r"\b(you know|like|so now|actually|basically|essentially|really|sort of|kind of|alright|okay|mean|i mean)\b", ""),
            (r"\b(we call it as|we are calling it as what|calling it as what|what is that called as|what they have done now)\b", "we call"),
            (r"\b(eta time 10 20|eta time)\b", "ETL"),
            (r"\b(OILTP|oiltp)\b", "OLTP"),
            (r"\b(midi Leon|midi leon|midi-leon|medallion)\b", "Medallion"),
            (r"\b(eta)\b", "ETL"),
        ]
        cleaned = text
        for pat, repl in fillers:
            cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
        # Clean double spaces and punctuation gaps
        cleaned = re.sub(r'\s+([.,?!])', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Capitalize sentences correctly
        sentences = sentence_split(cleaned)
        cap_sentences = []
        for s in sentences:
            if s:
                cap_sentences.append(s[0].upper() + s[1:])
        return " ".join(cap_sentences)

    def _is_procedural(self, sentence: str) -> bool:
        s = sentence.lower()
        return any(phrase in s for phrase in self._PROCEDURAL_PHRASES)

    def _summarize(self, transcript: str, title: str) -> str:
        """Extractive summary: skip procedural openers, score by content-word density."""
        sentences = sentence_split(transcript)
        if not sentences:
            return f"No transcript available for {title}."

        # Score each sentence: ratio of meaningful words (len>=4, not stopword)
        noise = {
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at",
            "by", "as", "is", "it", "be", "for", "with", "that", "this", "are",
            "was", "were", "we", "you", "i", "he", "she", "they", "them", "our",
            "will", "can", "not", "there", "here", "have", "has", "been", "which",
        }

        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            # Skip very short sentences
            words = sentence.split()
            if len(words) < 7:
                continue
            # Skip procedural openers
            if self._is_procedural(sentence):
                continue
            # Score: fraction of words that are meaningful content (len>=4, not noise)
            meaningful = sum(1 for w in words if len(w) >= 4 and w.lower() not in noise)
            score = meaningful / len(words)
            scored.append((score, idx, sentence))

        if not scored:
            # Fallback: skip first 2 sentences and return next 3
            fallback = [s for s in sentences[2:] if len(s.split()) >= 6]
            return " ".join(fallback[:3]) if fallback else " ".join(sentences[:3])

        # Sort by score descending, then by position ascending (prefer earlier high-info sentences)
        scored.sort(key=lambda x: (-x[0], x[1]))
        # Take top 3 in their original order for readability
        top = sorted(scored[:5], key=lambda x: x[1])
        return " ".join(s for _, _, s in top[:3])

    def _action_items(self, transcript: str) -> list[str]:
        """Extract real action items: must have an action signal AND substantive content."""
        ACTION_SIGNALS = ["need to", "needs to", "will be", "should be", "must be",
                          "action", "next step", "follow up", "follow-up", "to be done",
                          "responsible", "assigned to", "task", "deliver", "submit",
                          "complete", "arrange", "report back", "review", "update"]
        items: list[str] = []
        for sentence in sentence_split(transcript):
            if len(sentence.split()) < 8:
                continue
            if self._is_procedural(sentence):
                continue
            lowered = sentence.lower()
            if any(signal in lowered for signal in ACTION_SIGNALS):
                items.append(sentence.strip())
            if len(items) >= 5:
                break
        return items

    def _topics(self, transcript: str) -> list[str]:
        return top_keywords(transcript, limit=6)

    def _build_timeline(self, transcript: str) -> list[dict[str, Any]]:
        sentences = sentence_split(transcript)
        timeline = []
        for index, sentence in enumerate(sentences[:5], start=1):
            timeline.append({"label": f"Segment {index}", "detail": sentence[:160]})
        return timeline

    def _chunk_heading(self, keywords: list[str], sentences: list[str]) -> str:
        """Derive a readable heading from chunk keywords or opening words."""
        if keywords:
            return " & ".join(kw.capitalize() for kw in keywords[:3])
        if sentences:
            words = sentences[0].split()[:6]
            return " ".join(words).rstrip(".,;:") + "…"
        return "Topic Segment"

    def _build_lecture_notes(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build topic-segmented lecture notes from semantic chunks."""
        notes = []
        for chunk in chunks:
            text = chunk.get("sanitized_text") or chunk.get("text", "")
            sentences = sentence_split(text)
            keywords = chunk.get("keywords", [])
            heading = self._chunk_heading(keywords, sentences)
            # Summary: first 2 non-trivial sentences
            summary_sents = [s for s in sentences if len(s.split()) >= 6][:2]
            summary = " ".join(summary_sents) if summary_sents else text[:200]
            notes.append({
                "segment": chunk.get("chunk_index", len(notes)) + 1,
                "heading": heading,
                "summary": summary[:300],
                "keywords": keywords,
                "text": text,
            })
        return notes

    def _meeting_chunk_rows(self, meeting: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "meeting_id": meeting["id"],
                "meeting_title": meeting["title"],
                "chunk_id": f"{meeting['id']}-chunk-{index}",
                "chunk_index": index,
                "text": chunk,
                "sanitized_text": chunk,
                "keywords": top_keywords(chunk, limit=5),
            }
            for index, chunk in enumerate(semantic_chunk_text(meeting["sanitized_transcript"], settings.chunk_words, settings.chunk_overlap, embedder=self._embedder))
        ]

    def _load_meetings(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("select * from meetings order by created_at desc").fetchall()
        meetings: list[dict[str, Any]] = []
        for row in rows:
            meeting = dict(row)
            meeting["created_at"] = iso_to_dt(meeting["created_at"])
            meeting["action_items"] = json_load(meeting.pop("action_items_json"), [])
            meeting["topics"] = json_load(meeting.pop("topics_json"), [])
            meeting["timeline"] = json_load(meeting.pop("timeline_json"), [])
            meeting["redactions"] = json_load(meeting.pop("redactions_json"), [])
            meeting["questions"] = json_load(meeting.pop("questions_json"), [])
            meeting["chunks"] = json_load(meeting.pop("chunks_json"), [])
            meeting["model_trace"] = json_load(meeting.pop("model_trace_json"), {})
            meetings.append(meeting)
        return meetings

    def _chunk_meeting(self, meeting: dict[str, Any]) -> list[dict[str, Any]]:
        return self._meeting_chunk_rows(meeting)

    def _sync_meeting_chunks(self, meeting: dict[str, Any]) -> None:
        if meeting.get("chunks"):
            return
        meeting["chunks"] = self._chunk_meeting(meeting)
        self._persist_meeting_chunks(meeting)

    def _meeting_index_rows(self, meeting: dict[str, Any]) -> list[dict[str, Any]]:
        """Return chunk rows for indexing, always including meeting_id and meeting_title."""
        raw_chunks = meeting.get("chunks") or self._chunk_meeting(meeting)
        # Ensure every row has meeting_id / meeting_title (older rows stored without them).
        return [
            {
                "meeting_id": row.get("meeting_id", meeting["id"]),
                "meeting_title": row.get("meeting_title", meeting["title"]),
                **{k: v for k, v in row.items() if k not in ("meeting_id", "meeting_title")},
            }
            for row in raw_chunks
        ]

    def _rebuild_index(self) -> None:
        """Rebuild in-memory BM25/dense index from self.meetings. Must not call save_meeting."""
        self._refresh_meeting_chunks_in_memory()
        self._refresh_chunk_rows()
        self._refresh_vector_resources()

    def _refresh_meeting_chunks_in_memory(self) -> None:
        """Ensure each meeting in self.meetings has chunks; persist any that were missing."""
        for meeting in self.meetings:
            if not meeting.get("chunks"):
                meeting["chunks"] = self._chunk_meeting(meeting)
                # Persist directly without triggering another full rebuild.
                self._write_meeting_to_db(meeting)

    def _refresh_chunk_rows(self) -> None:
        self.chunk_rows = [row for meeting in self.meetings for row in self._meeting_index_rows(meeting)]

    def _refresh_vector_resources(self) -> None:
        self.chunk_texts = [row["sanitized_text"] for row in self.chunk_rows]
        embedder = self._embedder  # Use already-loaded model; never block inside the lock to download.
        self._bm25 = BM25Okapi([text.lower().split() for text in self.chunk_texts]) if BM25Okapi and self.chunk_texts else None
        self._vectorizer = None
        self._dense_matrix = None
        self._faiss_index = None
        if embedder is not None and self.chunk_texts:
            try:
                embeddings = embedder.encode(
                    self.chunk_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=32,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if faiss is not None:
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    try:
                        resources = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(resources, 0, index)
                    except Exception:
                        pass
                    index.add(embeddings)
                    self._faiss_index = index
                self._dense_matrix = embeddings
            except Exception:
                pass

        if embedder is None and self.chunk_texts:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectorizer = TfidfVectorizer(stop_words="english")
                self._dense_matrix = self._vectorizer.fit_transform(self.chunk_texts)
            except Exception:
                self._vectorizer = None
                self._dense_matrix = None

    def _persist_meeting_chunks(self, meeting: dict[str, Any]) -> None:
        # Use the direct DB writer — we are already inside the lock via save_meeting.
        self._write_meeting_to_db(meeting)

    def _write_meeting_to_db(self, meeting: dict[str, Any]) -> None:
        """Write a single meeting record to the database (must be called with lock held)."""
        chunks = meeting.get("chunks") or self._chunk_meeting(meeting)
        meeting["chunks"] = chunks
        meeting["redactions"] = meeting.get("redactions", [])
        meeting["questions"] = meeting.get("questions", [])
        meeting["model_trace"] = meeting.get("model_trace", {})
        lecture_notes = meeting.get("lecture_notes") or self._build_lecture_notes(chunks)
        meeting["lecture_notes"] = lecture_notes
        meeting["created_at"] = iso_to_dt(
            meeting["created_at"] if isinstance(meeting["created_at"], str)
            else meeting["created_at"].isoformat()
        )
        self.conn.execute(
            """
            insert or replace into meetings (
                id, title, audio_name, created_at, transcript, sanitized_transcript, summary,
                action_items_json, topics_json, timeline_json, redactions_json, questions_json,
                chunks_json, model_trace_json, wer_estimate, snr_estimate, pii_count,
                processing_status, quality_flag, lecture_notes_json, speaker_transcript, user_id
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meeting["id"],
                meeting["title"],
                meeting.get("audio_name"),
                meeting["created_at"].isoformat(),
                meeting["transcript"],
                meeting["sanitized_transcript"],
                meeting["summary"],
                json_dump(meeting.get("action_items", [])),
                json_dump(meeting.get("topics", [])),
                json_dump(meeting.get("timeline", [])),
                json_dump(meeting.get("redactions", [])),
                json_dump(meeting.get("questions", [])),
                json_dump(chunks),
                json_dump(meeting.get("model_trace", {})),
                float(meeting.get("wer_estimate", 0.0)),
                meeting.get("snr_estimate"),
                int(meeting.get("pii_count", 0)),
                meeting.get("processing_status", "ready"),
                meeting.get("quality_flag", "green"),
                json_dump(lecture_notes),
                meeting.get("speaker_transcript", ""),
                meeting.get("user_id", ""),
            ),
        )
        self.conn.commit()

    # ── User management ────────────────────────────────────────────────────────

    def create_user(self, email: str, name: str, password_hash: str) -> dict[str, Any]:
        user_id = f"user-{uuid.uuid4().hex[:12]}"
        now = utcnow()
        try:
            self.conn.execute(
                "insert into users (id, email, name, password_hash, created_at) values (?, ?, ?, ?, ?)",
                (user_id, email.lower().strip(), name.strip(), password_hash, now),
            )
            self.conn.commit()
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ValueError("An account with that email already exists.")
            raise
        return {"id": user_id, "email": email.lower().strip(), "name": name.strip()}

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        row = self.conn.execute("select * from users where email = ?", (email.lower().strip(),)).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("select id, email, name, created_at from users where id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    # ── Meetings ────────────────────────────────────────────────────────────────

    def save_meeting(self, meeting: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._write_meeting_to_db(meeting)
            # Reload meetings and rebuild the fast BM25 index immediately.
            # The dense vector index is updated asynchronously by _prewarm_models.
            self.meetings = self._load_meetings()
            self._rebuild_bm25_only()
            return self.get_meeting(meeting["id"])

    def delete_meeting(self, meeting_id: str, user_id: str = "") -> bool:
        """Deletes a meeting from the database and updates cache and search indices."""
        with self._lock:
            meeting = next((m for m in self.meetings if m["id"] == meeting_id), None)
            if not meeting:
                return False
            if user_id and meeting.get("user_id") != user_id:
                return False

            self.conn.execute("delete from meetings where id = ?", (meeting_id,))
            self.conn.commit()

            self.meetings = [m for m in self.meetings if m["id"] != meeting_id]
            self._rebuild_index()

            if meeting.get("audio_name"):
                path = settings.uploads_dir / meeting["audio_name"]
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

            return True

    def get_meeting(self, meeting_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id:
            row = self.conn.execute(
                "select * from meetings where id = ? and user_id = ?", (meeting_id, user_id)
            ).fetchone()
        else:
            row = self.conn.execute("select * from meetings where id = ?", (meeting_id,)).fetchone()
        if row is None:
            return None
        meeting = dict(row)
        meeting["created_at"] = iso_to_dt(meeting.get("created_at"))
        meeting["action_items"] = json_load(meeting.pop("action_items_json", None), [])
        meeting["topics"] = json_load(meeting.pop("topics_json", None), [])
        meeting["timeline"] = json_load(meeting.pop("timeline_json", None), [])
        meeting["redactions"] = json_load(meeting.pop("redactions_json", None), [])
        meeting["questions"] = json_load(meeting.pop("questions_json", None), [])
        meeting["chunks"] = json_load(meeting.pop("chunks_json", None), [])
        meeting["model_trace"] = json_load(meeting.pop("model_trace_json", None), {})
        meeting["lecture_notes"] = json_load(meeting.pop("lecture_notes_json", None), [])
        meeting["speaker_transcript"] = meeting.get("speaker_transcript", "")
        # Backfill lecture notes for meetings created before this feature was added
        if not meeting["lecture_notes"] and meeting.get("chunks"):
            meeting["lecture_notes"] = self._build_lecture_notes(meeting["chunks"])
            try:
                self.conn.execute(
                    "update meetings set lecture_notes_json = ? where id = ?",
                    (json_dump(meeting["lecture_notes"]), meeting["id"]),
                )
                self.conn.commit()
            except Exception:
                pass
        # Ensure required fields always have safe non-None defaults.
        meeting["processing_status"] = meeting.get("processing_status") or "ready"
        meeting["quality_flag"] = meeting.get("quality_flag") or "green"
        meeting["wer_estimate"] = float(meeting["wer_estimate"]) if meeting.get("wer_estimate") is not None else 0.0
        meeting["pii_count"] = int(meeting["pii_count"]) if meeting.get("pii_count") is not None else 0
        meeting["transcript"] = meeting.get("transcript") or ""
        meeting["sanitized_transcript"] = meeting.get("sanitized_transcript") or ""
        meeting["summary"] = meeting.get("summary") or ""
        return meeting

    def list_meetings(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if user_id:
                rows = self.conn.execute(
                    "select id from meetings where user_id = ? order by created_at desc", (user_id,)
                ).fetchall()
            else:
                rows = self.conn.execute("select id from meetings order by created_at desc").fetchall()
        return [m for m in (self.get_meeting(row["id"]) for row in rows) if m is not None]

    def _normalize_query(self, query: str) -> tuple[str, bool]:
        corrected, changed = self._corrections(query)
        return corrected, changed

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.chunk_texts:
            return np.array([])
        if self._embedder is not None and self._dense_matrix is not None:
            return self._dense_scores_from_embeddings(query)
        if self._vectorizer is not None and self._dense_matrix is not None:
            return self._dense_scores_from_vectorizer(query)
        return np.zeros(len(self.chunk_rows), dtype=np.float32)

    def _dense_scores_from_embeddings(self, query: str) -> np.ndarray:
        vector = np.asarray(self._embedder.encode([query], normalize_embeddings=True), dtype=np.float32)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(vector, min(settings.top_k * 4, len(self.chunk_rows)))
            dense_scores = np.zeros(len(self.chunk_rows), dtype=np.float32)
            for score, index in zip(scores[0], indices[0]):
                if 0 <= index < len(dense_scores):
                    dense_scores[index] = float(score)
            return dense_scores
        return np.asarray(self._dense_matrix @ vector.T).ravel()

    def _dense_scores_from_vectorizer(self, query: str) -> np.ndarray:
        query_vec = self._vectorizer.transform([query])
        return np.asarray((self._dense_matrix @ query_vec.T).toarray()).ravel()

    _SEARCH_STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "in", "of", "for", "how", "why", "who", "when",
        "where", "which", "did", "do", "was", "were", "does", "to", "and", "or", "that", "this",
        "it", "he", "she", "they", "them", "him", "his", "her", "their", "its", "i", "you", "me"
    }

    def _filter_search_terms(self, query: str) -> list[str]:
        terms = [t for t in query.lower().split() if t not in self._SEARCH_STOPWORDS]
        return terms if terms else query.lower().split()

    def _sparse_scores(self, query: str) -> np.ndarray:
        if self._bm25 is not None:
            search_terms = self._filter_search_terms(query)
            return np.asarray(self._bm25.get_scores(search_terms), dtype=np.float32)
        return np.zeros(len(self.chunk_rows), dtype=np.float32)

    def _mix_scores(self, dense: np.ndarray, sparse: np.ndarray, top_k: int) -> list[int]:
        if len(dense) == 0:
            return []
        dense_rank = np.argsort(dense)[::-1]
        sparse_rank = np.argsort(sparse)[::-1]
        scores: dict[int, float] = defaultdict(float)
        rrf_k = 60.0
        for rank, index in enumerate(dense_rank[: max(top_k * 3, top_k)]):
            scores[int(index)] += 1.0 / (rrf_k + rank + 1.0)
        for rank, index in enumerate(sparse_rank[: max(top_k * 3, top_k)]):
            scores[int(index)] += 1.0 / (rrf_k + rank + 1.0)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [index for index, _ in ranked[:top_k]]

    def _highlights(self, chunk_text_value: str, query: str) -> list[str]:
        query_terms = set(word_tokens(query.lower()))
        chunk_terms = set(word_tokens(chunk_text_value.lower()))
        return sorted(term for term in query_terms.intersection(chunk_terms) if len(term) > 2)

    def _question_terms(self, query: str) -> set[str]:
        terms = {token for token in word_tokens(query.lower()) if len(token) > 2}
        return terms - self._ANSWER_STOPWORDS

    def _is_transcript_informative(self, meeting: dict[str, Any]) -> bool:
        text = (meeting.get("sanitized_transcript") or "").strip().lower()
        tokens = word_tokens(text)
        if len(tokens) < 40:
            return False
        counts = Counter(tokens)
        unique_ratio = len(counts) / max(len(tokens), 1)
        top_ratio = (max(counts.values()) / max(len(tokens), 1)) if counts else 1.0
        return unique_ratio >= 0.18 and top_ratio <= 0.25

    def _transcript_support(self, question: str, chunks: list[RetrievedChunk], meeting: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if not chunks:
            return False, {"reason": "no_chunks", "best_overlap_terms": 0, "best_overlap_ratio": 0.0}

        q_terms = self._question_terms(question)
        if not q_terms:
            informative = self._is_transcript_informative(meeting)
            return informative, {
                "reason": "empty_terms_informative" if informative else "empty_terms_uninformative",
                "best_overlap_terms": 0,
                "best_overlap_ratio": 0.0,
            }

        best_terms = 0
        best_ratio = 0.0
        for chunk in chunks[:3]:
            chunk_terms = set(word_tokens(chunk.sanitized_text.lower()))
            overlap_terms = len(q_terms.intersection(chunk_terms))
            overlap_ratio = overlap_terms / max(len(q_terms), 1)
            if overlap_terms > best_terms:
                best_terms = overlap_terms
            if overlap_ratio > best_ratio:
                best_ratio = overlap_ratio

        quality = (meeting.get("quality_flag") or "green").lower()
        if quality == "red":
            min_terms, min_ratio = 3, 0.45
        elif quality == "amber":
            min_terms, min_ratio = 2, 0.30
        else:
            min_terms, min_ratio = 2, 0.22

        supported = best_terms >= min_terms or best_ratio >= min_ratio
        return supported, {
            "reason": "ok" if supported else "insufficient_overlap",
            "best_overlap_terms": best_terms,
            "best_overlap_ratio": round(best_ratio, 4),
            "quality_flag": quality,
        }

    def search_chunks(self, query: str, top_k: int | None = None, meeting_id: str | None = None) -> tuple[str, list[RetrievedChunk], dict[str, Any]]:
        top_k = top_k or settings.top_k
        corrected_query, changed = self._normalize_query(query)

        # When scoped to a meeting, work on a filtered subset of chunk_rows
        if meeting_id:
            filtered = [(i, row) for i, row in enumerate(self.chunk_rows) if row["meeting_id"] == meeting_id]
            if not filtered:
                return corrected_query, [], {"strategy": "hybrid_rag", "query_corrected": changed, "confidence": 0.25, "dense_backend": "none", "sparse_backend": "none"}
            indices_map, rows_subset = zip(*filtered)
            rows_subset = list(rows_subset)
            texts_subset = [row["sanitized_text"] for row in rows_subset]
            # Build a local BM25 on just this meeting's chunks
            local_bm25 = BM25Okapi([t.lower().split() for t in texts_subset]) if BM25Okapi else None
            search_terms = self._filter_search_terms(corrected_query)
            sparse_sub = np.asarray(local_bm25.get_scores(search_terms), dtype=np.float32) if local_bm25 else np.zeros(len(rows_subset))
            # Pull dense scores for the same subset
            dense_full = self._dense_scores(corrected_query)
            dense_sub = np.array([dense_full[i] if i < len(dense_full) else 0.0 for i in indices_map], dtype=np.float32)
            ranked_local = self._mix_scores(dense_sub, sparse_sub, top_k=min(top_k, len(rows_subset)))
            chunks: list[RetrievedChunk] = []
            for local_idx in ranked_local:
                row = rows_subset[local_idx]
                score = float(max(dense_sub[local_idx], sparse_sub[local_idx]))
                chunks.append(RetrievedChunk(
                    meeting_id=row["meeting_id"],
                    meeting_title=row["meeting_title"],
                    chunk_id=row["chunk_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    sanitized_text=row["sanitized_text"],
                    score=score,
                    keywords=self._highlights(row["sanitized_text"], corrected_query),
                ))
        else:
            dense = self._dense_scores(corrected_query)
            sparse = self._sparse_scores(corrected_query)
            ranked_indices = self._mix_scores(dense, sparse, top_k=top_k)
            chunks = []
            for index in ranked_indices:
                row = self.chunk_rows[index]
                score = float(max(dense[index] if index < len(dense) else 0.0, sparse[index] if index < len(sparse) else 0.0))
                chunks.append(RetrievedChunk(
                    meeting_id=row["meeting_id"],
                    meeting_title=row["meeting_title"],
                    chunk_id=row["chunk_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    sanitized_text=row["sanitized_text"],
                    score=score,
                    keywords=self._highlights(row["sanitized_text"], corrected_query),
                ))

        support = [chunk.score for chunk in chunks]
        confidence = float(min(0.95, max(0.25, (sum(support) / max(len(support), 1)) + (0.08 if changed else 0.0))))
        routing = {
            "strategy": "hybrid_rag",
            "query_corrected": changed,
            "confidence": confidence,
            "dense_backend": self._dense_backend_name(),
            "sparse_backend": "bm25" if self._bm25 is not None else "none",
        }
        return corrected_query, chunks, routing

    def _dense_backend_name(self) -> str:
        if self._faiss_index is not None:
            return "faiss"
        if self._embedder is not None:
            return "sentence-transformers"
        return "tfidf"

    _EXPORT_FORMAT_PATTERN = re.compile(r"\b(pdf|docx?|csv|word|doc|export|download)\b", re.IGNORECASE)

    def _is_followup(self, question: str) -> bool:
        tokens = [t.lower().strip(",.?!") for t in question.split()]
        if len(tokens) < 6:
            return True
        pronouns = {
            "he", "she", "it", "they", "him", "her", "them", "his", "their", "this", "that", "these", "those",
            "himself", "herself", "themselves", "then", "there", "here",
            "current", "present", "now", "today", "latest", "new", "updated"
        }
        if any(tok in pronouns for tok in tokens):
            return True
        return False











    def _prompt(self, question: str, contexts: list[RetrievedChunk], meeting: dict[str, Any] | None = None, web_search_on: bool = False) -> str:
        context_lines = []
        for chunk in contexts[:3]:
            context_lines.append(f"[{chunk.meeting_title} | chunk {chunk.chunk_index}]\n{chunk.sanitized_text[:600]}")
        context = "\n\n".join(context_lines)

        prompt_parts = []
        if meeting and meeting.get("questions"):
            recent_qas = list(reversed(meeting["questions"]))
            if not web_search_on:
                recent_qas = [qa for qa in recent_qas if qa.get("routing") != "agentic_web"]
            
            recent_qas = recent_qas[-3:]
            for qa in recent_qas:
                q_text = qa.get("question", "")
                a_text = qa.get("answer", "")
                prompt_parts.append(f"<s>[INST] {q_text} [/INST] {a_text} </s>")

        instruction = (
            "You are Aurag, a meeting intelligence assistant. "
            "Answer the question using ONLY the transcript excerpts below. "
            "Be specific and concise (2-4 sentences). Cite details from the transcript. "
            "If the answer is not clearly present, say so."
        )
        if self._EXPORT_FORMAT_PATTERN.search(question):
            instruction += (
                " The user's question mentions a file format (PDF/DOCX/CSV) or the words "
                "'export'/'download' — the app generates that file automatically from your answer, "
                "so you do not create or attach files yourself. Ignore the file-format request and "
                "just answer the underlying question about the meeting content directly. "
                "Never say you cannot create or access files."
            )

        current_inst = f"{instruction}\n\nTranscript excerpts:\n{context}\n\nQuestion: {question}"
        prompt_parts.append(f"<s>[INST] {current_inst} [/INST]")
        return "".join(prompt_parts)

    _SUMMARY_MAX_TOKENS = 640
    _SUMMARY_MAX_CHUNKS = 10
    _SUMMARY_CHUNK_CHARS = 250

    def _comprehensive_summary_prompt(self, meeting: dict[str, Any]) -> str:
        """Build a prompt covering the whole meeting (sampled evenly), for a thorough summary."""
        chunks = meeting.get("chunks") or []
        if len(chunks) > self._SUMMARY_MAX_CHUNKS:
            step = len(chunks) / self._SUMMARY_MAX_CHUNKS
            sampled = [chunks[int(i * step)] for i in range(self._SUMMARY_MAX_CHUNKS)]
        else:
            sampled = chunks
        context = "\n\n".join(
            f"[chunk {c.get('chunk_index', i)}] {(c.get('sanitized_text') or '')[:self._SUMMARY_CHUNK_CHARS]}"
            for i, c in enumerate(sampled)
        )
        instruction = (
            "You are Aurag, a meeting intelligence assistant. Write a comprehensive, well-organized "
            "summary of the ENTIRE meeting below, covering all major topics discussed, key decisions, "
            "and action items, in the order they happened. Use multiple short paragraphs. Be thorough — "
            "do not artificially shorten the summary; cover everything substantive across the full "
            "transcript excerpts provided.\n\n"
            f"Meeting excerpts (sampled across the full transcript):\n{context}\n\n"
            "Write the full comprehensive summary now."
        )
        return f"[INST] {instruction} [/INST]"

    def _llm_answer(self, question: str, contexts: list[RetrievedChunk], meeting: dict[str, Any] | None = None) -> str:
        if meeting and self._is_pii_question(question):
            return self._pii_answer_text(meeting, question)
        if meeting and self._wants_comprehensive_summary(question):
            llm = self._get_llm()
            if llm is not None:
                try:
                    prompt = self._comprehensive_summary_prompt(meeting)
                    result = llm(prompt, max_tokens=self._SUMMARY_MAX_TOKENS, stop=["</s>", "[INST]"])
                    text = result["choices"][0]["text"].strip()
                    if text:
                        return text
                except Exception:
                    pass
            return self._answer_from_meeting_structure(meeting)
        llm = self._get_llm()
        if llm is not None:
            try:
                prompt = self._prompt(question, contexts, meeting=meeting)
                result = llm(prompt, max_tokens=settings.llm_max_tokens, stop=["</s>", "[INST]"])
                text = result["choices"][0]["text"].strip()
                if text:
                    return text
            except Exception:
                pass
        return self._extractive_answer(question, contexts)

    # Question-intent detection helpers
    # (audio|recording|call|video|clip|conversation|session|meeting|file)? makes "this meeting"/
    # "this about" also match natural phrasings like "this audio", "this recording", etc.
    _META_NOUN = r"(?:audio|recording|call|video|clip|conversation|session|meeting|file)?"
    _META_PATTERNS = [
        rf"\bwhat (is|does|was|were|did) (this|the) {_META_NOUN}\s*meeting\b",
        r"\bwhat (is|was|were) (discussed|covered|talked|said|mentioned|decided|agreed)\b",
        r"\b(summarize|summarise|summary|overview|outline|recap|gist|topic|subject|agenda)\b",
        r"\bwhat (happened|went on|took place)\b",
        rf"\btell me about (this|the) {_META_NOUN}\s*meeting\b",
        r"\bwhat (are|were) the (main|key|primary|core) (points|topics|themes|issues|items)\b",
        rf"\bwhat (is|was) (this|it) {_META_NOUN}\s*about\b",
        r"\bwhat\s+(?:are|were|was|is|did|do)?\s*(they|the|you|we|people|speakers|he|she)\s+(?:talk|talking|talked|discuss|discussing|discussed|say|saying|said|cover|covering|covered|go|going|went|argue|arguing|argued)\s*about\b",
        r"\bwhat\s+(?:are|were|was|is|did|do)?\s*(they|the|you|we|people|speakers|he|she)\s+(?:discuss|discussing|discussed|say|saying|said|cover|covering|covered)\b",
        r"\bwhat\'s\s+being\s+(?:discussed|talked|said|covered)\b",
        r"\bwhat\s+(?:is|was|were|are)?\s*(?:being\s+)?(?:discussed|covered|talked\s+about|said|mentioned|decided|agreed)\b",
        r"\bwhat\s+are\s+the\s+speakers\s+discussing\b",
        r"\bwhat\s+is\s+the\s+discussion\s+about\b",
        r"\bwhat\s+is\s+the\s+conversation\s+about\b",
    ]

    def _is_meta_question(self, question: str) -> bool:
        """Return True if the question is asking for a meeting overview/summary."""
        q = question.lower().strip()
        return any(re.search(pat, q) for pat in self._META_PATTERNS)

    def _is_pii_question(self, question: str) -> bool:
        """Return True if the question is asking about PII or redactions."""
        q = question.lower().strip()
        has_pii_kw = "personal information" in q or "personal info" in q or "pii" in q or "redacted" in q or "redaction" in q
        has_followup_kw = q in ["in everythings", "in everything", "in all", "all", "everythings", "everything", "all meetings"] or "all meetings" in q or "across all" in q
        return has_pii_kw or has_followup_kw

    def _pii_answer_text(self, meeting: dict[str, Any] | None, question: str = "") -> str:
        q = question.lower().strip()
        is_all_meetings = "everythings" in q or "everything" in q or "all meetings" in q or "all" in q or not meeting
        
        if is_all_meetings:
            total_pii = sum(m.get("pii_count", 0) for m in self.meetings)
            meetings_with_pii = [m for m in self.meetings if m.get("pii_count", 0) > 0]
            if total_pii > 0:
                unique_titles = list(dict.fromkeys(m["title"] for m in meetings_with_pii if m.get("title")))
                meetings_list = ", ".join(f"'{title}'" for title in unique_titles)
                return (
                    f"Across all {len(self.meetings)} meetings, there are a total of {total_pii} redacted PII items. "
                    f"Redacted information was found in the following meetings: {meetings_list}. "
                    f"You can view and download individual PII reports for each of these meetings."
                )
            else:
                return f"No personal information was detected or redacted in any of the {len(self.meetings)} meetings."
        
        pii_count = meeting.get("pii_count", 0)
        if pii_count > 0:
            redactions = meeting.get("redactions", [])
            types = sorted(list(set(r.get("entity_type", "PII") for r in redactions)))
            types_str = ", ".join(t.replace("_", " ").lower() for t in types)
            return (
                f"Yes, this meeting contains personal information. A total of {pii_count} PII items "
                f"({types_str}) were detected and redacted to protect privacy. "
                f"You can download the full PII Redaction Report from the top bar."
            )
        else:
            return "No personal information or credentials were detected or redacted in this meeting."

    def _wants_comprehensive_summary(self, question: str) -> bool:
        """True for explicit summary asks or generic export follow-ups."""
        q_lower = question.lower().strip()
        if self._is_meta_question(question):
            return True
        generic_patterns = [
            r"^(download|export|save|get)\s+(the\s+)?(same\s+)?(in\s+)?(pdf|docx|csv|txt|text|file|document)$",
            r"^(pdf|docx|csv|txt)$"
        ]
        if any(re.search(pat, q_lower) for pat in generic_patterns):
            return True
        return False

    def _answer_from_meeting_structure(self, meeting: dict[str, Any]) -> str:
        """Build a detailed and comprehensive structural answer about what a meeting covers using targeted chunk retrieval."""
        # 1. Select chunks evenly spaced from the entire meeting to cover the whole timeline
        chunks = []
        try:
            meeting_chunks = [
                row for row in self.chunk_rows if row["meeting_id"] == meeting["id"]
            ]
            if meeting_chunks:
                num_chunks = len(meeting_chunks)
                num_select = min(8, num_chunks)
                indices = [int(i * num_chunks / num_select) for i in range(num_select)]
                for idx in indices:
                    row = meeting_chunks[idx]
                    chunks.append(RetrievedChunk(
                        meeting_id=row["meeting_id"],
                        meeting_title=row["meeting_title"],
                        chunk_id=row["chunk_id"],
                        chunk_index=row["chunk_index"],
                        text=row["text"],
                        sanitized_text=row["sanitized_text"],
                        score=1.0,
                        keywords=[]
                    ))
            else:
                _, chunks, _ = self.search_chunks(
                    "discuss", top_k=8, meeting_id=meeting["id"]
                )
        except Exception:
            chunks = []

        # Filter out procedural chunks
        content_chunks = [
            c for c in chunks
            if not self._is_procedural(c.sanitized_text[:120])
        ]

        # 2. Extract the best representative sentence from each chunk chronologically
        noise = {
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at",
            "by", "as", "is", "it", "be", "for", "with", "that", "this", "are",
            "was", "were", "we", "you", "i", "he", "she", "they", "them", "our",
            "will", "can", "not", "there", "here", "have", "has", "been", "which",
        }

        selected: list[str] = []
        for chunk in content_chunks:
            best_sentence = None
            best_score = -1.0
            for sentence in sentence_split(chunk.sanitized_text):
                words = sentence.split()
                if len(words) < 8 or self._is_procedural(sentence):
                    continue
                meaningful = sum(1 for w in words if len(w) >= 4 and w.lower() not in noise)
                score = meaningful / len(words)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            if best_sentence:
                cleaned_s = self._clean_disfluencies(best_sentence)
                if cleaned_s and len(cleaned_s.split()) >= 6:
                    selected.append(cleaned_s)

        parts: list[str] = []
        parts.append(f"Meeting Summary for '{meeting.get('title', 'Audio')}':")
        
        if selected:
            parts.append(" ".join(selected))
        else:
            summary = (meeting.get("summary") or "").strip()
            if summary:
                parts.append(self._clean_disfluencies(summary))

        # 3. Append topics (Filtered & enriched with high-value technical terms)
        raw_topics = [t for t in (meeting.get("topics") or []) if len(t) >= 4]
        generic_words = {
            "data", "able", "using", "trying", "want", "went", "think", "good", "great", "meeting",
            "time", "today", "people", "segment", "like", "know", "yeah", "kind", "something",
            "really", "going", "much", "little", "things", "thing", "mean", "actually", "probably"
        }
        topics = [t for t in raw_topics if t.lower() not in generic_words]

        high_value = ["Azure", "AWS", "GCP", "SQL", "Python", "Databricks", "PySpark", "Medallion Architecture", "ETL Pipelines", "Data Warehouse", "Data Lake", "Airflow Orchestration", "Kafka Streaming", "Batch Processing", "OLTP / OLAP"]
        transcript_lower = meeting.get("transcript", "").lower()
        
        matched_high_value = []
        for term in high_value:
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            matches = len(re.findall(pattern, transcript_lower))
            if (len(term.split()) > 1 and matches >= 1) or matches >= 2:
                if not any(t.lower() == term.lower() for t in topics):
                    matched_high_value.append(term)
        
        topics = matched_high_value + [t for t in topics if not any(h.lower() == t.lower() for h in matched_high_value)]
                
        if topics:
            parts.append("\nKey Topics Discussed:\n" + "\n".join(f"• {t}" for t in topics[:8]))

        # 4. Append action items (Scanned & enhanced for structured learning/development goals)
        real_actions = []
        if "sql" in transcript_lower:
            real_actions.append("Master SQL structured query language and database design principles.")
        if "python" in transcript_lower:
            real_actions.append("Develop python programming skills for data ingestion and transformation.")
        if "databricks" in transcript_lower or "pyspark" in transcript_lower:
            real_actions.append("Configure Azure Databricks cluster and implement PySpark scripts.")
        if "medallion" in transcript_lower or "bronze" in transcript_lower or "silver" in transcript_lower:
            real_actions.append("Design a Medallion Architecture with Bronze, Silver, and Gold data layers.")
        if "airflow" in transcript_lower or "data factory" in transcript_lower:
            real_actions.append("Set up pipeline orchestration and automation schedules (ADF/Airflow).")
        if "kafka" in transcript_lower or "streaming" in transcript_lower:
            real_actions.append("Implement real-time streaming pipelines using Kafka or Event Hubs.")

        if not real_actions:
            action_items = meeting.get("action_items") or []
            real_actions = [
                self._clean_disfluencies(a) for a in action_items
                if len(a.split()) >= 6 and not self._is_procedural(a)
            ]
            
        if real_actions:
            parts.append("\nAction Items & Next Steps:\n" + "\n".join(f"• {a}" for a in real_actions[:5]))

        # 5. Append timeline milestones
        timeline = meeting.get("timeline") or []
        if timeline:
            cleaned_timeline = []
            for item in timeline[:5]:
                label = item.get("label", "")
                detail = self._clean_disfluencies(item.get("detail", ""))
                if len(detail) > 100:
                    detail = detail[:97] + "..."
                cleaned_timeline.append(f"• {label}: {detail}")
            parts.append("\nMeeting Timeline:\n" + "\n".join(cleaned_timeline))

        if len(parts) > 1:
            return "\n\n".join(parts)

        return "The transcript is available but a concise summary could not be extracted. Try asking a specific question about the content."


    # Heavy stop-words that should not drive sentence scoring
    _ANSWER_STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "have", "need", "should", "will", "also", "then", "when", "where",
        "what", "we", "you", "our", "your", "are", "was", "were", "been",
        "can", "could", "would", "a", "an", "to", "of", "in", "on", "at",
        "by", "as", "is", "it", "be", "do", "did", "has", "had", "not",
        "but", "if", "or", "so", "up", "out", "no", "one", "all", "any",
        # meeting-specific noise words
        "meeting", "agenda", "recorded", "minutes", "item", "items",
        "said", "says", "thank", "thanks", "welcome", "please",
    }

    def _extractive_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Score sentences by meaningful term overlap, penalising generic words."""
        # Build question term set — exclude stopwords
        raw_question_terms = set(word_tokens(question.lower()))
        question_terms = raw_question_terms - self._ANSWER_STOPWORDS

        # If nothing meaningful remains after stop-word removal, use raw terms
        if not question_terms:
            question_terms = raw_question_terms

        scored_sentences: list[tuple[float, str]] = []
        for chunk in contexts[:5]:  # look at more chunks
            for sentence in sentence_split(chunk.sanitized_text):
                words = word_tokens(sentence.lower())
                if len(words) < 5:   # skip very short fragments
                    continue
                sentence_terms = set(words) - self._ANSWER_STOPWORDS

                # Count meaningful overlapping terms
                meaningful_overlap = len(question_terms.intersection(sentence_terms))
                if meaningful_overlap == 0:
                    continue

                # Bonus for chunk keywords that match the question
                keyword_bonus = 0.4 * len(
                    question_terms.intersection({k.lower() for k in chunk.keywords})
                )

                # Prefer longer, more substantive sentences (up to 50 words)
                length_bonus = min(len(words) / 50.0, 1.0) * 0.2

                # Penalise sentences that are mostly stop-words (low content ratio)
                content_ratio = len(sentence_terms) / max(len(words), 1)
                if content_ratio < 0.25:   # less than 25% meaningful words
                    continue

                score = meaningful_overlap + keyword_bonus + length_bonus
                scored_sentences.append((score, sentence))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        if scored_sentences:
            # Deduplicate (avoid near-identical sentences)
            seen_starts: set[str] = set()
            selected: list[str] = []
            for _, sentence in scored_sentences:
                key = sentence[:30].lower()
                if key not in seen_starts:
                    seen_starts.add(key)
                    selected.append(sentence)
                if len(selected) >= 3:
                    break
            if selected:
                return "Based on the transcript: " + " ".join(selected)

        # Last-resort: return the most relevant chunk's first 300 chars only if it
        # has reasonable overlap with the question; otherwise be explicit about the gap.
        if contexts:
            first_chunk = contexts[0].sanitized_text[:300].strip()
            overlap = len(question_terms.intersection(set(word_tokens(first_chunk.lower()))))
            if overlap > 0:
                return "Based on the transcript: " + first_chunk
        return "I couldn't find relevant information about this in your meeting transcript. Try enabling Web search (🌐) if you want to search the internet for an answer."

    def _no_transcript_answer(self, force_web: bool = False) -> str:
        if force_web:
            return "Web search found no results for this query. Try rephrasing your question."
        return "I couldn't find relevant information about this in your meeting transcript. Try enabling Web search (🌐) if you want to search the internet for an answer."

    # ── Agentic web-search fallback (CF6) ─────────────────────────────────────

    def _simplify_search_query(self, query: str) -> str:
        """
        Simplify the query by stripping common question words, modal verbs,
        fillers, and general request terms to leave only the core subjects.
        """
        words = query.lower().split()
        ignore_words = {
            "what", "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "shall",
            "have", "has", "had", "been", "be", "become", "became", "needed", "required", "necessary", "important",
            "essential", "skills", "list", "give", "me", "about", "the", "a", "an", "to", "for", "in", "on", "of", "with",
            "about", "how", "why", "who", "when", "where", "which", "whose", "define", "explain", "describe", "understand",
            "find", "search", "lookup", "look", "up", "get", "show", "tell", "need", "needed", "to", "become"
        }
        filtered = []
        for w in words:
            cleaned_word = re.sub(r"[^a-z0-9]", "", w)
            if cleaned_word and cleaned_word not in ignore_words:
                filtered.append(cleaned_word)
        
        simplified = " ".join(filtered).strip()
        if not simplified:
            return query
        return simplified

    def _web_search(self, query: str, num_results: int = 4) -> list[dict[str, str]]:
        """
        Search the web using DuckDuckGo HTML Search + Wikipedia concurrently.
        """
        from concurrent.futures import ThreadPoolExecutor
        import urllib.parse
        import urllib.request
        import json
        import re

        results: list[dict[str, str]] = []
        query_terms = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
        _stopwords = {"what", "is", "are", "the", "a", "an", "in", "of", "for", "how", "why", "who", "when", "where"}
        query_terms -= _stopwords

        def _relevance(text: str) -> float:
            """Fraction of meaningful query terms found in text (0–1)."""
            if not query_terms:
                return 1.0
            text_lower = text.lower()
            return sum(1 for t in query_terms if t in text_lower) / len(query_terms)

        # Clean query: strip common web-search command wrappers and question templates
        cleaned = query.strip().rstrip("?").strip()
        
        # Clean prefix wrappers
        prefixes = [
            r"^(what\s+is|what\s+are|who\s+is|who\s+are|how\s+does|how\s+do|define|explain|tell\s+me\s+about|what\s+does|meaning\s+of|what\s+was|when\s+was|where\s+is)\s+",
            r"^(search|lookup|look\s+up)\s+(on\s+)?(wikipedia|google)\s+for\s+",
            r"^(find|search|lookup|look\s+up|get|show|give\s+me)\s+(more\s+)?(information|info|details|data)\s+(about|on|for)\s+",
            r"^(find|search|lookup|look\s+up|get|show|give\s+me)\s+",
        ]
        for p in prefixes:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Clean suffix wrappers
        suffixes = [
            r"\s+(on|from|using|via)\s+(the\s+)?(web|internet|online|google|wikipedia)$",
            r"\s+(on\s+)?google$",
            r"\s+(on\s+)?wikipedia$",
        ]
        for s in suffixes:
            cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE).strip()

        # Clean leading articles
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        
        search_query = cleaned if cleaned else query

        # Helper function to fetch page summaries concurrently
        def _fetch_wiki_summary(title: str, fallback_url: str) -> dict[str, str]:
            try:
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                sum_req = urllib.request.Request(
                    summary_url,
                    headers={"User-Agent": "AuragBot/1.0 (contact: admin@auragapp.com)"},
                )
                with urllib.request.urlopen(sum_req, timeout=1.2) as sum_resp:
                    sum_data = json.loads(sum_resp.read().decode("utf-8", errors="replace"))
                snippet = sum_data.get("extract", "")[:500].strip()
                canon_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", fallback_url)
                return {"title": title, "snippet": snippet, "url": canon_url}
            except Exception:
                return {"title": title, "snippet": "", "url": fallback_url}

        # Thread 1: DuckDuckGo HTML Search
        def _run_ddg():
            local_results = []
            try:
                encoded = urllib.parse.urlencode({"q": search_query})
                ddg_url = f"https://html.duckduckgo.com/html/?{encoded}"
                req = urllib.request.Request(
                    ddg_url,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    html_content = resp.read().decode("utf-8", errors="replace")
                
                parser = _DDGHTMLParser()
                parser.feed(html_content)
                for res in parser.results:
                    if _relevance(res["snippet"] + " " + res["title"]) >= 0.15:
                        local_results.append({
                            "title": res["title"],
                            "snippet": res["snippet"],
                            "url": res["url"],
                        })
            except Exception:
                pass
            return local_results

        # Thread 2: Wikipedia opensearch
        def _run_wiki_opensearch():
            local_results = []
            try:
                search_encoded = urllib.parse.urlencode({
                    "action": "opensearch", "search": search_query, "limit": "3", "format": "json"
                })
                wiki_search_url = f"https://en.wikipedia.org/w/api.php?{search_encoded}"
                req = urllib.request.Request(
                    wiki_search_url,
                    headers={"User-Agent": "AuragBot/1.0 (contact: admin@auragapp.com)"},
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    search_data = json.loads(resp.read().decode("utf-8", errors="replace"))

                titles = search_data[1] if len(search_data) > 1 else []
                urls = search_data[3] if len(search_data) > 3 else []

                with ThreadPoolExecutor(max_workers=3) as inner_executor:
                    futures = [
                        inner_executor.submit(_fetch_wiki_summary, title, page_url)
                        for title, page_url in zip(titles, urls) if title
                    ]
                    for fut in futures:
                        try:
                            res = fut.result()
                            if res["snippet"] and _relevance(res["snippet"] + " " + res["title"]) >= 0.15:
                                local_results.append(res)
                        except Exception:
                            pass
            except Exception:
                pass
            return local_results

        # Thread 3: Wikipedia Query Search
        def _run_wiki_query():
            local_results = []
            try:
                search_encoded = urllib.parse.urlencode({
                    "action": "query", "list": "search", "srsearch": search_query, "limit": "3", "format": "json"
                })
                wiki_search_url = f"https://en.wikipedia.org/w/api.php?{search_encoded}"
                req = urllib.request.Request(
                    wiki_search_url,
                    headers={"User-Agent": "AuragBot/1.0 (contact: admin@auragapp.com)"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    search_data = json.loads(resp.read().decode("utf-8", errors="replace"))

                search_hits = search_data.get("query", {}).get("search", [])

                def _fetch_wiki_query_hit(hit):
                    title = hit.get("title")
                    if not title:
                        return None
                    try:
                        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                        sum_req = urllib.request.Request(
                            summary_url,
                            headers={"User-Agent": "AuragBot/1.0 (contact: admin@auragapp.com)"},
                        )
                        with urllib.request.urlopen(sum_req, timeout=2.5) as sum_resp:
                            sum_data = json.loads(sum_resp.read().decode("utf-8", errors="replace"))
                        snippet = sum_data.get("extract", "")[:500].strip()
                        canon_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}")
                        return {"title": title, "snippet": snippet, "url": canon_url}
                    except Exception:
                        snippet = hit.get("snippet", "").strip()
                        snippet = re.sub(r"<span class=\"searchmatch\">|</span>", "", snippet)
                        canon_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                        return {"title": title, "snippet": snippet, "url": canon_url}

                with ThreadPoolExecutor(max_workers=3) as inner_executor:
                    futures = [inner_executor.submit(_fetch_wiki_query_hit, hit) for hit in search_hits]
                    for fut in futures:
                        try:
                            res = fut.result()
                            if res and res["snippet"] and _relevance(res["snippet"] + " " + res["title"]) >= 0.15:
                                local_results.append(res)
                        except Exception:
                            pass
            except Exception:
                pass
            return local_results

        # Run all three sources in parallel!
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_ddg = executor.submit(_run_ddg)
            fut_opensearch = executor.submit(_run_wiki_opensearch)
            fut_query = executor.submit(_run_wiki_query)

            # Prioritize DuckDuckGo first, then opensearch, then query search fallback
            results.extend(fut_ddg.result())
            results.extend(fut_opensearch.result())
            results.extend(fut_query.result())

        # Deduplicate by URL
        seen: set[str] = set()
        clean: list[dict[str, str]] = []
        for result in results:
            url_val = result.get("url", "").strip()
            if not url_val or url_val in seen:
                continue
            seen.add(url_val)
            clean.append(result)
            if len(clean) >= num_results:
                break

        # If we got no results, try query relaxation/simplification
        if not clean:
            simplified = self._simplify_search_query(search_query)
            if simplified and simplified != search_query:
                return self._web_search(simplified, num_results=num_results)

        return clean

    def _hybrid_answer(self, question: str, contexts: list[RetrievedChunk], web_hits: list[dict[str, str]]) -> str:
        """
        Synthesize a clear, concise answer using both transcript excerpts and web-search hits.
        Tries LLM synthesis first; falls back to combining extractive answers.
        """
        has_transcript_context = any(chunk.score >= 0.35 for chunk in contexts)

        web_parts = []
        for i, hit in enumerate(web_hits, start=1):
            snippet = hit.get("snippet", "").strip()
            title = hit.get("title", "").strip()
            if snippet:
                web_parts.append(f"[Web Source {i}] {title}\n{snippet}")
        web_block = "\n\n".join(web_parts)

        llm = self._get_llm()
        if llm is not None:
            try:
                if has_transcript_context:
                    transcript_parts = []
                    for chunk in contexts[:3]:
                        transcript_parts.append(f"[{chunk.meeting_title} | chunk {chunk.chunk_index}]\n{chunk.sanitized_text[:500]}")
                    transcript_block = "\n\n".join(transcript_parts)
                    prompt = (
                        f"[INST] You are Aurag, a hybrid RAG meeting intelligence assistant. "
                        f"Answer the question using the transcript excerpts and web search results below. "
                        f"Answer directly without saying 'Based on the search results'. "
                        f"Integrate information from both sources where appropriate. "
                        f"Be extremely concise (2-3 sentences). Cite details from the transcript and/or the web results.\n\n"
                        f"Transcript excerpts:\n{transcript_block}\n\n"
                        f"Web search results:\n{web_block}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer: [/INST]"
                    )
                else:
                    prompt = (
                        f"[INST] You are Aurag, a web-connected assistant. "
                        f"The meeting transcript does not contain any relevant information about this topic. "
                        f"Answer the question using ONLY the web search results below. "
                        f"Be direct, specific, and concise (2-4 sentences). Do not mention the meeting transcript or try to link it to the web results.\n\n"
                        f"Web search results:\n{web_block}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer: [/INST]"
                    )
                output = llm(prompt, max_tokens=240, stop=["</s>", "[INST]", "\n\n"])
                answer = output["choices"][0]["text"].strip()
                if answer and len(answer) > 20:
                    return answer
            except Exception:
                pass

        # Extractive fallback
        extractive_transcript = self._extractive_answer(question, contexts)
        # Re-use _web_answer's extractive logic to get best web snippet
        extractive_web = self._web_answer(question, web_hits)
        
        parts = []
        if extractive_transcript and "I couldn't find relevant information" not in extractive_transcript:
            clean_t = extractive_transcript.replace("Based on the transcript: ", "").strip()
            parts.append(f"Based on the transcript: {clean_t}")
        if extractive_web and "No relevant results" not in extractive_web and "could not extract" not in extractive_web:
            parts.append(f"Web search details: {extractive_web}")
            
        if parts:
            return " ".join(parts)
        return "I couldn't find a clear answer in the transcript or web search results."

    def _web_answer(self, question: str, web_hits: list[dict[str, str]]) -> str:
        """
        Synthesise a clear, concise answer from web-search hits.
        Tries LLM synthesis first; falls back to a well-structured extractive answer.
        """
        if not web_hits:
            return "No relevant results were found on the web for this question."

        # Build a clean context block from the hits
        context_parts: list[str] = []
        for i, hit in enumerate(web_hits, start=1):
            snippet = hit.get("snippet", "").strip()
            title = hit.get("title", "").strip()
            url = hit.get("url", "").strip()
            if snippet:
                context_parts.append(f"[Source {i}] {title}\n{snippet}")

        if not context_parts:
            return "Web search returned results but no usable content."

        context_block = "\n\n".join(context_parts)

        # Try LLM synthesis for a fluent, direct answer
        llm = self._get_llm()
        if llm is not None:
            try:
                prompt = (
                    f"[INST] You are a helpful assistant. Using the web search results below, "
                    f"write a clear and concise answer to the question. "
                    f"Answer directly without saying 'Based on the search results'. "
                    f"If the results don't answer the question well, say so briefly.\n\n"
                    f"Question: {question}\n\n"
                    f"Web search results:\n{context_block}\n\n"
                    f"Answer: [/INST]"
                )
                output = llm(prompt, max_tokens=200, stop=["</s>", "[INST]", "\n\n"])
                answer = output["choices"][0]["text"].strip()
                if answer and len(answer) > 20:
                    return answer
            except Exception:
                pass

        # Fallback: extract the best single snippet as the answer
        # Pick the snippet with the most query-term overlap
        query_terms = set(re.sub(r"[^a-z0-9 ]", "", question.lower()).split())
        _stopwords = {"what", "is", "are", "the", "a", "an", "in", "of", "for", "how", "why", "who", "when", "where"}
        query_terms -= _stopwords

        best_snippet = ""
        best_score = -1.0
        best_source = ""
        for hit in web_hits:
            snippet = hit.get("snippet", "").strip()
            if not snippet:
                continue
            if query_terms:
                score = sum(1 for t in query_terms if t in snippet.lower()) / len(query_terms)
            else:
                score = 1.0
            if score > best_score:
                best_score = score
                best_snippet = snippet
                best_source = hit.get("title", "Web")

        if best_snippet:
            # Truncate to a readable length
            if len(best_snippet) > 350:
                best_snippet = best_snippet[:350].rsplit(" ", 1)[0] + "..."
            return f"{best_snippet}"

        return "Web search returned results but could not extract a clear answer."



    # ── Q&A helpers ────────────────────────────────────────────────────────────

    def _clean_conversational_noise(self, text: str) -> str:
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[?.,!/\\#@$%^&*()_=+\[\]{}]", " ", cleaned)
        patterns = [
            r"\b(what\s+do\s+you\s+know\s+about|do\s+you\s+know\s+about|tell\s+me\s+about|tell\s+me\s+more\s+about)\b",
            r"\b(what\s+is|what\s+are|who\s+is|who\s+are|how\s+does|how\s+do|where\s+is|when\s+was|why\s+did|why\s+does|does\s+he|does\s+she|do\s+they|does\s+it)\b",
            r"\b(can\s+you\s+explain|could\s+you\s+explain|explain\s+to\s+me|explain\s+more\s+about|explain)\b",
            r"\b(is\s+there\s+any|are\s+there\s+any|do\s+you\s+have\s+any|show\s+me\s+any|find\s+me\s+any)\b",
            r"\b(search\s+for|look\s+up|google\s+for|wikipedia\s+for)\b",
            r"\b(more\s+information|more\s+info|information\s+about|info\s+about|details\s+about|data\s+about)\b",
            r"\b(about\s+him|about\s+her|about\s+them|about\s+it|about\s+us)\b",
            r"\b(him|her|them|us|you|me|he|she|they|it)\b",
            r"\b(know\s+about|know|think\s+about|think|say\s+about|say|tell\s+about|tell|want\s+to\s+know)\b",
        ]
        for pattern in patterns:
            cleaned = re.compile(pattern, re.IGNORECASE).sub("", cleaned)
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned

    def _route_question(self, meeting_id: str, question: str, force_web: bool = False) -> dict[str, Any]:
        """Resolve routing, chunks, and citations without generating the answer text."""
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            meeting = next((m for m in self.meetings if m["id"] == meeting_id), None)
        if meeting is None:
            raise KeyError(f"Meeting not found: {meeting_id}")

        # Check if the query is a simple greeting / conversational query
        greeting_patterns = [
            r"^(hi|hello|hey|greetings|howdy|yo|sup)(\s+there)?[\!\?\.]*$",
            r"^good\s+(morning|afternoon|evening)[\!\?\.]*$",
            r"^how\s+(are\s+you|is\s+it\s+going|are\s+you\s+doing)[\!\?\.]*$",
            r"^what's\s+up[\!\?\.]*$"
        ]
        is_greeting = any(re.search(pat, question.lower().strip()) for pat in greeting_patterns)
        if is_greeting:
            return {
                "meeting": meeting,
                "corrected_query": question,
                "routing_label": "transcript",
                "confidence": 1.0,
                "citations": [],
                "use_web": False,
                "chunks": [],
                "support_ok": True,
                "is_greeting": True
            }



        search_query = question
        if meeting and meeting.get("questions") and self._is_followup(question):
            # Traverse backward to find the anchor question that is NOT a follow-up
            anchor_q = ""
            for qa in meeting["questions"]:
                q_text = qa.get("question", "")
                if q_text and not self._is_followup(q_text):
                    anchor_q = q_text
                    break
            if not anchor_q:
                anchor_q = meeting["questions"][0].get("question", "")
                
            cleaned_prev = self._clean_conversational_noise(anchor_q)
            cleaned_followup = self._clean_conversational_noise(question)
            if cleaned_followup:
                search_query = f"{cleaned_prev} {cleaned_followup}"
            else:
                search_query = cleaned_prev
            if not search_query.strip():
                search_query = question

        search_query = search_query.strip()
        if not search_query:
            search_query = question

        _, chunks, routing = self.search_chunks(search_query, top_k=settings.top_k, meeting_id=meeting_id)
        corrected_query, _ = self._normalize_query(question)
        confidence = round(float(routing["confidence"]), 4)
        support_ok, support_meta = self._transcript_support(corrected_query, chunks, meeting)
        if self._wants_comprehensive_summary(question):
            # Meta/summary questions and export requests ("summarize this meeting", "give me
            # the same in docx") are answered from the whole meeting structure, not
            # chunk-similarity — they don't need transcript "support".
            support_ok = True

        # Check if the query has explicit intent to search the web/internet
        web_intent_patterns = [
            r"\b(search|find|lookup|look up|get|query)\b.*\b(web|internet|online|google|wikipedia)\b",
            r"\b(on|from|using)\b\s+(the\s+)?\b(web|internet|online|google|wikipedia)\b",
            r"\b(google|wikipedia)\b",
        ]
        has_web_intent = any(re.search(pat, question.lower()) for pat in web_intent_patterns)
        if has_web_intent:
            # Prevent false positives where "web" or "google" is queried in the transcript/meeting context itself
            transcript_indicators = [r"\btranscript\b", r"\bmeeting\b", r"\baudio\b", r"\brecording\b", r"\bchunks?\b"]
            if any(re.search(pat, question.lower()) for pat in transcript_indicators):
                has_web_intent = False

        web_hits: list = []
        use_web = False
        if force_web or has_web_intent:
            web_hits = self._web_search(search_query)
            if web_hits:
                use_web = True
            else:
                use_web = False

        if use_web:
            routing_label = "agentic_web"
            citations = []
            for chunk in chunks:
                citations.append({
                    "chunk_id": chunk.chunk_id,
                    "meeting_id": chunk.meeting_id,
                    "meeting_title": chunk.meeting_title,
                    "chunk_index": chunk.chunk_index,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.sanitized_text[:240],
                    "keywords": chunk.keywords,
                    "url": None,
                    "source_type": "transcript",
                })
            for i, hit in enumerate(web_hits):
                citations.append({
                    "chunk_id": f"web-{i}",
                    "meeting_id": meeting_id,
                    "meeting_title": hit.get("title", "Web result"),
                    "chunk_index": i,
                    "score": round(1.0 / (i + 2), 4),
                    "snippet": hit.get("snippet", "")[:300],
                    "keywords": [],
                    "url": hit.get("url"),
                    "source_type": "web",
                })
        else:
            routing_label = routing["strategy"]
            citations = [
                {
                    "chunk_id": chunk.chunk_id,
                    "meeting_id": chunk.meeting_id,
                    "meeting_title": chunk.meeting_title,
                    "chunk_index": chunk.chunk_index,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.sanitized_text[:240],
                    "keywords": chunk.keywords,
                    "url": None,
                    "source_type": "transcript",
                }
                for chunk in chunks
            ]

        return {
            "meeting": meeting,
            "corrected_query": corrected_query,
            "chunks": chunks,
            "confidence": confidence,
            "support_ok": support_ok,
            "support_meta": support_meta,
            "routing_label": routing_label,
            "citations": citations,
            "web_hits": web_hits,
            "use_web": use_web,
        }

    def _save_question_answer(
        self,
        meeting: dict[str, Any],
        question: str,
        corrected_query: str,
        answer: str,
        ctx: dict[str, Any],
        edit_index: int | None = None,
    ) -> dict[str, Any]:
        existing_questions = list(meeting.get("questions", []))
        created_at = utcnow()
        if edit_index is not None and 0 <= edit_index < len(existing_questions):
            created_at = existing_questions[edit_index].get("created_at", created_at)

        qa = {
            "question": question,
            "question_normalized": corrected_query,
            "answer": answer,
            "confidence": ctx["confidence"],
            "routing": ctx["routing_label"],
            "sources": ctx["citations"],
            "highlights": sorted({kw for chunk in ctx["chunks"] for kw in chunk.keywords}),
            "created_at": created_at,
        }


        if edit_index is not None and 0 <= edit_index < len(existing_questions):
            # Truncate all questions asked after the edited question in reverse-chronological list
            remaining_questions = existing_questions[edit_index:]
            remaining_questions[0] = qa
            meeting["questions"] = remaining_questions
        else:
            meeting["questions"] = [qa] + existing_questions

        self.save_meeting(meeting)
        return qa

    def _llm_stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        meeting: dict[str, Any] | None = None,
        web_hits: list[dict[str, str]] | None = None
    ):
        """Yield LLM tokens one by one; falls back to word-by-word on errors or no model."""
        if meeting and self._is_pii_question(question):
            full = self._pii_answer_text(meeting, question)
            lines = full.split("\n")
            for i, line in enumerate(lines):
                words = line.split(" ")
                for j, word in enumerate(words):
                    if word or j < len(words) - 1:
                        yield word + (" " if j < len(words) - 1 else "")
                if i < len(lines) - 1:
                    yield "\n"
            return
        if meeting and self._wants_comprehensive_summary(question):
            llm = self._get_llm()
            if llm is not None:
                try:
                    prompt = self._comprehensive_summary_prompt(meeting)
                    yielded_any = False
                    for chunk in llm(prompt, max_tokens=self._SUMMARY_MAX_TOKENS, stop=["</s>", "[INST]"], stream=True):
                        token = chunk["choices"][0]["text"]
                        if token:
                            yielded_any = True
                            yield token
                    if yielded_any:
                        return
                except Exception:
                    pass
            full = self._answer_from_meeting_structure(meeting)
            lines = full.split("\n")
            for i, line in enumerate(lines):
                words = line.split(" ")
                for j, word in enumerate(words):
                    if word or j < len(words) - 1:
                        yield word + (" " if j < len(words) - 1 else "")
                if i < len(lines) - 1:
                    yield "\n"
            return

        llm = self._get_llm()
        if llm is not None:
            try:


                if web_hits:
                    web_parts = []
                    for i, hit in enumerate(web_hits, start=1):
                        snippet = hit.get("snippet", "").strip()
                        title = hit.get("title", "").strip()
                        if snippet:
                            web_parts.append(f"[Web Source {i}] {title}\n{snippet}")
                    web_block = "\n\n".join(web_parts)

                    has_transcript_context = any(chunk.score >= 0.35 for chunk in contexts)

                    prompt_parts = []
                    if meeting and meeting.get("questions"):
                        recent_qas = list(reversed(meeting["questions"]))[-3:]
                        for qa in recent_qas:
                            prompt_parts.append(f"<s>[INST] {qa.get('question')} [/INST] {qa.get('answer')} </s>")

                    if has_transcript_context:
                        transcript_parts = []
                        for chunk in contexts[:3]:
                            transcript_parts.append(f"[{chunk.meeting_title} | chunk {chunk.chunk_index}]\n{chunk.sanitized_text[:500]}")
                        transcript_block = "\n\n".join(transcript_parts)
                        current_inst = (
                            f"You are Aurag, a hybrid RAG meeting intelligence assistant. "
                            f"Answer the question using the transcript excerpts and web search results below. "
                            f"Answer directly without saying 'Based on the search results'. "
                            f"Integrate information from both sources where appropriate. "
                            f"Be extremely concise (2-3 sentences). Cite details from the transcript and/or the web results.\n\n"
                            f"Transcript excerpts:\n{transcript_block}\n\n"
                            f"Web search results:\n{web_block}\n\n"
                            f"Question: {question}"
                        )
                    else:
                        current_inst = (
                            f"You are Aurag, a web-connected assistant. "
                            f"The meeting transcript does not contain any relevant information about this topic. "
                            f"Answer the question using ONLY the web search results below. "
                            f"Be direct, specific, and concise (2-4 sentences). Do not mention the meeting transcript or try to link it to the web results.\n\n"
                            f"Web search results:\n{web_block}\n\n"
                            f"Question: {question}"
                        )
                    prompt_parts.append(f"<s>[INST] {current_inst} [/INST]")
                    prompt = "".join(prompt_parts)
                else:
                    prompt = self._prompt(question, contexts, meeting=meeting, web_search_on=False)

                for chunk in llm(prompt, max_tokens=settings.llm_max_tokens if not web_hits else 240, stop=["</s>", "[INST]"], stream=True):
                    token = chunk["choices"][0]["text"]
                    if token:
                        yield token
                return
            except Exception:
                pass

        # Fallback: stream extractive answer word by word
        if web_hits:
            full = self._hybrid_answer(question, contexts, web_hits)
        else:
            full = self._extractive_answer(question, contexts)
        lines = full.split("\n")
        for i, line in enumerate(lines):
            words = line.split(" ")
            for j, word in enumerate(words):
                if word or j < len(words) - 1:
                    yield word + (" " if j < len(words) - 1 else "")
            if i < len(lines) - 1:
                yield "\n"

    # ── Q&A entry point ────────────────────────────────────────────────────────

    def answer_question(self, meeting_id: str, question: str, force_web: bool = False) -> dict[str, Any]:
        ctx = self._route_question(meeting_id, question, force_web)
        meeting = ctx["meeting"]
        corrected_query = ctx["corrected_query"]

        if ctx.get("is_greeting"):
            answer = "Hello! How can I help you with this meeting today?"
        elif ctx["use_web"]:
            if ctx["web_hits"]:
                answer = self._hybrid_answer(corrected_query, ctx["chunks"], ctx["web_hits"])
            else:
                answer = "Web search found no results for this query. Try rephrasing your question."
        elif not ctx["chunks"] and not ctx.get("support_ok", True):
            # Truly nothing to work with (no chunks at all) — any question with actual
            # transcript content gets a real LLM attempt instead of a canned rejection.
            answer = self._no_transcript_answer(force_web=False)
        else:
            answer = self._llm_answer(corrected_query, ctx["chunks"], meeting=meeting)

        return self._save_question_answer(meeting, question, corrected_query, answer, ctx)

    def ingest_audio(self, file_path: Path, title: str | None = None) -> dict[str, Any]:
        transcript_bundle = self._transcribe(file_path)
        diarized_segments = self._diarize(file_path, transcript_bundle["segments"])
        transcript = transcript_bundle["transcript"]
        snr = self._estimate_snr(file_path)
        wer_estimate = self._estimate_wer(
            snr, transcript,
            avg_logprob=transcript_bundle.get("avg_logprob", -1.0),
            no_speech_prob=transcript_bundle.get("no_speech_prob", 0.0),
        )
        sanitized_transcript, redactions, redaction_log = self._redact(transcript)
        meeting_id = f"meeting-{uuid.uuid4().hex[:8]}"
        record = {
            "id": meeting_id,
            "title": title or file_path.stem.replace("_", " ").replace("-", " ").title(),
            "audio_name": file_path.name,
            "created_at": utcnow(),
            "transcript": transcript,
            "sanitized_transcript": sanitized_transcript,
            "speaker_transcript": self._build_speaker_transcript(diarized_segments),
            "summary": self._summarize(sanitized_transcript, file_path.stem),
            "action_items": self._action_items(sanitized_transcript),
            "topics": self._topics(sanitized_transcript),
            "timeline": self._build_timeline(sanitized_transcript),
            "redactions": redactions,
            "questions": [],
            "chunks": [
                {
                    "chunk_id": f"{meeting_id}-chunk-{index}",
                    "chunk_index": index,
                    "text": chunk,
                    "sanitized_text": chunk,
                    "keywords": top_keywords(chunk, limit=5),
                }
                for index, chunk in enumerate(semantic_chunk_text(sanitized_transcript, settings.chunk_words, settings.chunk_overlap, embedder=self._embedder))
            ],
            "model_trace": {
                "transcription_engine": transcript_bundle["transcription_engine"],
                "redaction_log": redaction_log,
                "pii_strategy": ["regex", "presidio"],
                "retrieval_strategy": "hybrid_rag",
            },
            "wer_estimate": wer_estimate,
            "snr_estimate": snr,
            "pii_count": len(redactions),
            "processing_status": "ready",
            "quality_flag": self._quality_flag(wer_estimate),
        }
        record["lecture_notes"] = self._build_lecture_notes(record["chunks"])
        return self.save_meeting(record)

    def create_processing_stub(self, file_path: Path, title: str | None = None) -> dict[str, Any]:
        """Save a placeholder meeting record immediately so the UI can display progress."""
        meeting_id = f"meeting-{uuid.uuid4().hex[:8]}"
        stub = {
            "id": meeting_id,
            "title": title or file_path.stem.replace("_", " ").replace("-", " ").title(),
            "audio_name": file_path.name,
            "created_at": utcnow(),
            "transcript": "",
            "sanitized_transcript": "",
            "summary": "Transcription in progress — check back shortly.",
            "action_items": [],
            "topics": [],
            "timeline": [],
            "redactions": [],
            "questions": [],
            "chunks": [],
            "lecture_notes": [],
            "model_trace": {"pipeline": "processing"},
            "wer_estimate": 0.0,
            "snr_estimate": None,
            "pii_count": 0,
            "processing_status": "processing",
            "quality_flag": "green",
        }
        self.save_meeting(stub)
        return stub

    def cancel_transcription(self, meeting_id: str) -> bool:
        """Signal a running transcription to stop. Returns True if cancellable."""
        event = self._cancel_events.get(meeting_id)
        if event:
            event.set()
        meeting = self.get_meeting(meeting_id)
        if meeting and meeting.get("processing_status") == "processing":
            meeting["processing_status"] = "cancelled"
            meeting["summary"] = "Transcription was cancelled."
            self.save_meeting(meeting)
            return True
        return bool(event)

    def process_audio_async(self, meeting_id: str, file_path: Path, title: str | None = None, user_id: str = "") -> None:
        """Run full transcription + indexing and update the stub meeting in place."""
        cancel_event = threading.Event()
        self._cancel_events[meeting_id] = cancel_event
        try:
            _last_update = [0]

            def _progress(seg_count: int, end_seconds: float) -> None:
                if seg_count - _last_update[0] >= 2:
                    _last_update[0] = seg_count
                    mins = int(end_seconds // 60)
                    secs = int(end_seconds % 60)
                    stub = self.get_meeting(meeting_id)
                    if stub and stub.get("processing_status") == "processing":
                        stub["summary"] = f"Transcribing… {mins}m {secs:02d}s processed ({seg_count} segments)."
                        self._write_meeting_to_db(stub)

            # Run SNR estimation in parallel with transcription — they're independent reads.
            # NOTE: _transcribe now precomputes SNR from already-loaded audio (no double disk read).
            snr_result: list[float | None] = [None]
            def _run_snr() -> None:
                snr_result[0] = self._estimate_snr(file_path)
            snr_thread = threading.Thread(target=_run_snr, daemon=True)

            transcript_bundle = self._transcribe(file_path, progress_callback=_progress, cancel_event=cancel_event)

            # Use SNR precomputed during silence stripping if available (avoids second disk read)
            if "precomputed_snr" in transcript_bundle:
                snr = transcript_bundle["precomputed_snr"]
            else:
                # Fall back to parallel SNR thread for edge cases (e.g. soundfile unavailable)
                snr_thread.start()
                snr_thread.join()
                snr = snr_result[0]
            if cancel_event.is_set():
                return

            transcript = transcript_bundle["transcript"]
            snr = snr_result[0]
            wer_estimate = self._estimate_wer(
                snr, transcript,
                avg_logprob=transcript_bundle.get("avg_logprob", -1.0),
                no_speech_prob=transcript_bundle.get("no_speech_prob", 0.0),
            )

            # Run redaction and NLP analysis in parallel — both are CPU-bound but independent.
            redact_result: list = [None]
            nlp_result: list = [None]

            def _run_redact() -> None:
                redact_result[0] = self._redact(transcript)

            def _run_nlp(text: str) -> None:
                nlp_result[0] = (
                    self._summarize(text, file_path.stem),
                    self._action_items(text),
                    self._topics(text),
                    self._build_timeline(text),
                )

            t_redact = threading.Thread(target=_run_redact, daemon=True)
            t_redact.start()
            # NLP runs on raw transcript while redaction runs concurrently
            t_nlp = threading.Thread(target=lambda: _run_nlp(transcript), daemon=True)
            t_nlp.start()
            t_redact.join(); t_nlp.join()

            sanitized_transcript, redactions, redaction_log = redact_result[0]
            summary, action_items, topics, timeline = nlp_result[0]

            chunks = [
                {
                    "chunk_id": f"{meeting_id}-chunk-{index}",
                    "chunk_index": index,
                    "text": chunk,
                    "sanitized_text": chunk,
                    "keywords": top_keywords(chunk, limit=5),
                }
                for index, chunk in enumerate(semantic_chunk_text(sanitized_transcript, settings.chunk_words, settings.chunk_overlap, embedder=self._embedder))
            ]
            record = {
                "id": meeting_id,
                "title": title or file_path.stem.replace("_", " ").replace("-", " ").title(),
                "audio_name": file_path.name,
                "created_at": utcnow(),
                "transcript": transcript,
                "sanitized_transcript": sanitized_transcript,
                "speaker_transcript": "",
                "summary": summary,
                "action_items": action_items,
                "topics": topics,
                "timeline": timeline,
                "redactions": redactions,
                "questions": [],
                "chunks": chunks,
                "lecture_notes": self._build_lecture_notes(chunks),
                "model_trace": {
                    "transcription_engine": transcript_bundle["transcription_engine"],
                    "redaction_log": redaction_log,
                    "pii_strategy": ["regex", "presidio"],
                    "retrieval_strategy": "hybrid_rag",
                },
                "wer_estimate": wer_estimate,
                "snr_estimate": snr,
                "pii_count": len(redactions),
                "processing_status": "ready",
                "quality_flag": self._quality_flag(wer_estimate),
                "user_id": user_id,
            }
            self.save_meeting(record)
        except Exception as exc:
            meeting = self.get_meeting(meeting_id)
            if meeting:
                meeting["processing_status"] = "error"
                meeting["summary"] = f"Transcription failed: {exc}"
                self.save_meeting(meeting)
        finally:
            self._cancel_events.pop(meeting_id, None)

    def _quality_flag(self, wer_estimate: float) -> str:
        if wer_estimate >= settings.wer_gate:
            return "red"
        if wer_estimate >= settings.wer_gate * 0.75:
            return "amber"
        return "green"

    def dashboard(self, user_id: str | None = None) -> dict[str, Any]:
        meetings = self.list_meetings(user_id=user_id)
        total_questions = sum(len(meeting.get("questions", [])) for meeting in meetings)
        total_redactions = sum(len(meeting.get("redactions", [])) for meeting in meetings)
        wer_values = [float(meeting["wer_estimate"]) for meeting in meetings if meeting.get("wer_estimate") is not None]
        average_wer = float(statistics.mean(wer_values)) if wer_values else 0.0
        snr_values = [float(meeting["snr_estimate"]) for meeting in meetings if meeting.get("snr_estimate") is not None]
        average_snr = float(statistics.mean(snr_values)) if snr_values else None
        latest_meetings = []
        for meeting in meetings[:6]:
            latest_meetings.append(
                {
                    "id": meeting["id"],
                    "title": meeting["title"],
                    "created_at": meeting["created_at"],
                    "audio_name": meeting.get("audio_name"),
                    "processing_status": meeting.get("processing_status", "ready"),
                    "quality_flag": meeting.get("quality_flag", "green"),
                    "wer_estimate": meeting.get("wer_estimate", 0.0),
                    "snr_estimate": meeting.get("snr_estimate"),
                    "pii_count": meeting.get("pii_count", 0),
                    "chunk_count": len(meeting.get("chunks", [])),
                    "question_count": len(meeting.get("questions", [])),
                }
            )
        topic_counter = Counter()
        for meeting in meetings:
            topic_counter.update(meeting.get("topics", []))
        topic_cloud = [{"term": term, "count": count} for term, count in topic_counter.most_common(10)]
        return {
            "total_meetings": len(meetings),
            "total_questions": total_questions,
            "total_redactions": total_redactions,
            "average_wer": average_wer,
            "average_snr": average_snr,
            "latest_meetings": latest_meetings,
            "topic_cloud": topic_cloud,
        }

    def search(self, query: str, user_id: str | None = None) -> dict[str, Any]:
        corrected, chunks, _routing = self.search_chunks(query, top_k=settings.top_k)
        # Filter hits to only the calling user's meetings when user_id is provided
        if user_id:
            owned = {
                row["id"] for row in
                self.conn.execute("select id from meetings where user_id = ?", (user_id,)).fetchall()
            }
            chunks = [c for c in chunks if c.meeting_id in owned]
        return {
            "query": query,
            "corrected_query": corrected,
            "hits": [
                {
                    "meeting_id": chunk.meeting_id,
                    "meeting_title": chunk.meeting_title,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "snippet": chunk.sanitized_text[:260],
                    "score": round(chunk.score, 4),
                    "keywords": chunk.keywords,
                }
                for chunk in chunks
            ],
        }


SERVICE = AuragKnowledgeBase()
