import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, ArrowUpRight } from 'lucide-react'
import Atmosphere from './Atmosphere'

/**
 * Shared layout for section pages — matches Welcome atmosphere + glass header.
 */
export default function PageShell({
  title,
  eyebrow = "Kostas' Dashboard",
  backTo = '/',
  backLabel = 'Home',
  onBack,
  children,
  contentClassName = '',
  maxWidthClass = 'max-w-5xl',
}) {
  const backClassName =
    'inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-emerald-900/10 bg-white/80 px-3 py-1.5 text-sm font-medium text-emerald-900 transition hover:border-whatsapp-500/40 hover:text-whatsapp-500'

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden bg-[#f3faf5] font-sans text-slate-800">
      <Atmosphere />

      <header className="relative z-30 sticky top-0 flex h-[70px] items-center justify-between border-b border-emerald-900/5 bg-white/70 px-5 backdrop-blur-md sm:px-8">
        <div className="flex min-w-0 items-center gap-3">
          <Link
            to="/"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-whatsapp-500 text-sm font-bold text-white shadow-sm shadow-whatsapp-500/30 transition hover:brightness-110"
            aria-label="Home"
          >
            K
          </Link>
          <div className="min-w-0">
            <p className="truncate text-[11px] font-semibold uppercase tracking-[0.16em] text-whatsapp-500">
              {eyebrow}
            </p>
            <h1 className="truncate text-base font-semibold text-emerald-950 sm:text-lg">
              {title}
            </h1>
          </div>
        </div>

        {onBack ? (
          <button type="button" onClick={onBack} className={backClassName}>
            <ArrowLeft className="h-3.5 w-3.5" strokeWidth={2.5} />
            {backLabel}
          </button>
        ) : (
          <Link to={backTo} className={backClassName}>
            <ArrowLeft className="h-3.5 w-3.5" strokeWidth={2.5} />
            {backLabel}
          </Link>
        )}
      </header>

      <main
        className={`relative z-10 mx-auto flex w-full flex-1 flex-col px-5 py-10 sm:px-8 sm:py-14 ${maxWidthClass} ${contentClassName}`}
      >
        {children}
      </main>
    </div>
  )
}

/** Welcome-style section intro. */
export function PageIntro({ eyebrow, title, description }) {
  return (
    <div className="mb-10 max-w-xl animate-welcome-rise sm:mb-12">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
        {eyebrow}
      </p>
      <h2 className="font-display text-4xl font-semibold leading-[1.15] tracking-tight text-emerald-950 sm:text-5xl">
        {title}
      </h2>
      {description && (
        <p className="mt-4 text-base leading-relaxed text-slate-600 sm:text-lg">
          {description}
        </p>
      )}
    </div>
  )
}

/** Glass panel that flattens nested tool card chrome. */
export function ToolPanel({ children, className = '' }) {
  return (
    <div
      className={`rounded-2xl border border-emerald-900/8 bg-white/80 p-4 shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm sm:p-6 [&>*]:mx-auto [&>*]:mt-0 [&>*]:max-w-none [&>*]:rounded-xl [&>*]:border-0 [&>*]:bg-transparent [&>*]:p-0 [&>*]:shadow-none ${className}`}
    >
      {children}
    </div>
  )
}

/** Hub card button matching the welcome page. */
export function HubCard({ label, description, icon: Icon, onClick, delayMs = 0 }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ animationDelay: `${delayMs}ms` }}
      className="group relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-emerald-900/8 bg-white/80 p-5 text-left shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm transition duration-300 animate-welcome-rise hover:-translate-y-0.5 hover:border-whatsapp-500/35 hover:bg-white hover:shadow-[0_12px_32px_-12px_rgba(45,188,58,0.35)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-whatsapp-500"
    >
      <div className="flex items-start justify-between gap-3">
        <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-whatsapp-100 to-emerald-50 text-whatsapp-500 ring-1 ring-emerald-900/5 transition duration-300 group-hover:from-whatsapp-500 group-hover:to-emerald-600 group-hover:text-white group-hover:ring-transparent">
          <Icon className="h-5 w-5" strokeWidth={2} />
        </span>
        <ArrowUpRight className="h-4 w-4 text-emerald-900/20 transition duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-whatsapp-500" />
      </div>
      <div>
        <div className="text-base font-semibold text-emerald-950 transition group-hover:text-whatsapp-500">
          {label}
        </div>
        <div className="mt-1.5 text-sm leading-snug text-slate-500">
          {description}
        </div>
      </div>
    </button>
  )
}
