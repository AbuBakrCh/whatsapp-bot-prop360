import React from 'react'
import { Link } from 'react-router-dom'
import {
  Wrench,
  Link2,
  Building2,
  BookOpen,
  Scale,
  Users,
  MessageCircle,
  ArrowUpRight,
} from 'lucide-react'
import { isMessengerEnabled } from './config/messenger'

const sections = [
  {
    to: '/utilities',
    label: 'Utilities',
    description: 'Email, contacts, properties, jobs, and more',
    icon: Wrench,
  },
  {
    to: '/important-links',
    label: 'Important links',
    description: 'Quick access to external tools',
    icon: Link2,
  },
  {
    to: '/spitogatos',
    label: 'Spitogatos',
    description: 'Property listings and crawler',
    icon: Building2,
  },
  {
    to: '/ledger',
    label: 'Ledger',
    description: 'Property cashflow and finances',
    icon: BookOpen,
  },
  {
    to: '/cashflow-matching',
    label: 'Accrual–Payment Matching',
    description: 'Match cashflow accruals to equal-amount payments by period',
    icon: Scale,
  },
  {
    to: '/groups',
    label: 'Groups',
    description: 'Manage and share with contact groups',
    icon: Users,
  },
]

const messengerSection = {
  to: '/messenger',
  label: 'WhatsApp Messenger',
  description: 'View and reply to WhatsApp conversations',
  icon: MessageCircle,
}

export default function App() {
  const visibleSections = isMessengerEnabled()
    ? [messengerSection, ...sections]
    : sections

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden bg-[#f3faf5] font-sans text-slate-800">
      {/* Atmosphere */}
      <div
        className="pointer-events-none absolute inset-0"
        aria-hidden="true"
      >
        <div className="absolute -top-32 -left-24 h-[28rem] w-[28rem] rounded-full bg-whatsapp-500/15 blur-3xl" />
        <div className="absolute top-1/3 -right-20 h-[24rem] w-[24rem] rounded-full bg-emerald-400/10 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-[18rem] w-[36rem] rounded-full bg-teal-300/10 blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.35]"
          style={{
            backgroundImage:
              'radial-gradient(circle at 1px 1px, rgba(15, 118, 70, 0.12) 1px, transparent 0)',
            backgroundSize: '24px 24px',
          }}
        />
      </div>

      <header className="relative z-30 sticky top-0 flex h-[70px] items-center justify-between border-b border-emerald-900/5 bg-white/70 px-5 backdrop-blur-md sm:px-8">
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-whatsapp-500 text-sm font-bold text-white shadow-sm shadow-whatsapp-500/30">
            K
          </span>
          <span className="text-sm font-semibold tracking-wide text-emerald-800 sm:text-base">
            Kostas&apos; Dashboard
          </span>
        </div>
        <span className="rounded-md bg-emerald-50 px-2 py-1 text-[11px] font-medium tabular-nums text-emerald-700/70">
          v{__APP_VERSION__}
        </span>
      </header>

      <main className="relative z-10 mx-auto flex w-full max-w-5xl flex-1 flex-col px-5 py-12 sm:px-8 sm:py-16">
        <div className="mb-12 max-w-xl animate-welcome-rise sm:mb-14">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
            Home
          </p>
          <h1 className="font-display text-4xl font-semibold leading-[1.15] tracking-tight text-emerald-950 sm:text-5xl">
            Kostas&apos; Dashboard
          </h1>
          <p className="mt-4 text-base leading-relaxed text-slate-600 sm:text-lg">
            Your workspace for messaging, utilities, properties, and finances.
            Pick a section to continue.
          </p>
        </div>

        <nav
          className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
          aria-label="Dashboard sections"
        >
          {visibleSections.map(({ to, label, description, icon: Icon }, index) => (
            <Link
              key={to}
              to={to}
              style={{ animationDelay: `${80 + index * 55}ms` }}
              className="group relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-emerald-900/8 bg-white/80 p-5 shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm transition duration-300 animate-welcome-rise hover:-translate-y-0.5 hover:border-whatsapp-500/35 hover:bg-white hover:shadow-[0_12px_32px_-12px_rgba(45,188,58,0.35)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-whatsapp-500"
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
            </Link>
          ))}
        </nav>
      </main>
    </div>
  )
}
