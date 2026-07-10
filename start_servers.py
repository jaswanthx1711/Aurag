import subprocess
import time
import re
import os
import sys

def kill_port(port):
    try:
        pid = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
        if pid:
            for p in pid.split():
                subprocess.run(["kill", "-9", p])
            print(f"Cleared port {port}")
    except Exception:
        pass

def main():
    print("🚀 Starting AuRAG Meeting Intelligence Platform...")
    
    # 1. Kill old processes to prevent port conflicts
    kill_port(3000)
    kill_port(8000)
    
    # 2. Start Backend
    print("🤖 Starting FastAPI Backend (Port 8000)...")
    backend_dir = os.path.abspath("backend")
    venv_python = os.path.join(os.path.abspath(".venv"), "bin", "uvicorn")
    subprocess.Popen(
        [venv_python, "app.main:app", "--port", "8000", "--reload"],
        cwd=backend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    # 3. Start Frontend
    print("🎨 Starting Next.js Production Frontend (Port 3000)...")
    frontend_dir = os.path.abspath("nextjs-frontend")
    subprocess.Popen(
        ["npm", "run", "start", "--", "-p", "3000"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    # 4. Start Cloudflare Tunnel
    print("🌐 Starting Cloudflare Secure Public Tunnel...")
    log_file = "cloudflared.log"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    with open(log_file, "w") as f:
        subprocess.Popen(
            ["npx", "--yes", "cloudflared", "tunnel", "--url", "http://localhost:3000"],
            stdout=f,
            stderr=f
        )
        
    # 5. Extract Cloudflare URL
    print("⏳ Waiting for Cloudflare URL allocation...")
    url = None
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", content)
                if match:
                    url = match.group(0)
                    break
                    
    if url:
        print("\n" + "="*50)
        print("🎉 AuRAG is Live & Publicly Accessible!")
        print(f"👉 Public Link: {url}")
        print("="*50)
        print("\nKeep this terminal window open during your presentation.")
    else:
        print("\n⚠️ Tunnel started but URL took too long to load. Check 'cloudflared.log' to find the URL.")

if __name__ == "__main__":
    main()
