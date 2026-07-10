import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AuRAG · Meeting Intelligence',
  description: 'AI-powered meeting intelligence platform',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">{children}</body>
    </html>
  )
}
