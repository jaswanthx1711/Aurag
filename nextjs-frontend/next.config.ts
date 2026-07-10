import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["thirty-monkeys-train.loca.lt", "*.loca.lt", "localhost:3000"],
  experimental: {
    // Audio uploads proxied through /api/* to the backend can exceed the 10MB default.
    proxyClientMaxBodySize: "500mb",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
