import React from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft,
  ArrowUpRight,
  Mail,
  FileText,
  FileUp,
  Copy,
  MessageCircle,
  Building2,
  UserPlus,
  Sparkles,
  ClipboardList,
  Calendar,
  Wallet,
  GitMerge,
  PlayCircle,
  Timer,
} from 'lucide-react'
import SendBulkEmail from './SendBulkEmail'
import ProcessBankStatementsDrive from './ProcessBankStatementsDrive'
import ShowDuplicates from './ShowDuplicates'
import GenerateClientMessages from './GenerateClientMessages'
import AddProperties from './AddProperties'
import ManageTimetables from './ManageTimetables'
import ManageCashflows from './ManageCashflows'
import AddContacts from './AddContacts'
import CreateContactFromInvoice from './CreateContactFromInvoice'
import GenerateActivitySummaries from './GenerateActivitySummaries'
import PropertyActivitySummaries from './PropertyActivitySummaries'
import MergeContacts from './MergeContacts'
import JobControl from './JobControl'
import ExpiryJobsControl from './ExpiryJobsControl'

const tools = [
  {
    id: 'bulk-email',
    label: 'Send Bulk Email',
    description: 'Send emails from a Drive spreadsheet',
    icon: Mail,
    group: 'Messaging',
    render: () => <SendBulkEmail />,
  },
  {
    id: 'client-messages',
    label: 'Client Messages',
    description: 'Generate outreach for clients',
    icon: MessageCircle,
    group: 'Messaging',
    render: () => <GenerateClientMessages />,
  },
  {
    id: 'activity-summaries',
    label: 'Activity Summaries',
    description: 'Generate daily activity digests',
    icon: Sparkles,
    group: 'Messaging',
    render: () => <GenerateActivitySummaries />,
  },
  {
    id: 'property-summaries',
    label: 'Property Summaries',
    description: 'Activity summaries by property',
    icon: ClipboardList,
    group: 'Messaging',
    render: () => <PropertyActivitySummaries />,
  },
  {
    id: 'process-docs',
    label: 'Process Documents',
    description: 'Bank statements and Drive files',
    icon: FileText,
    group: 'Data',
    render: () => <ProcessBankStatementsDrive />,
  },
  {
    id: 'duplicates',
    label: 'Duplicate Contacts',
    description: 'Find and review contact overlaps',
    icon: Copy,
    group: 'Data',
    render: () => <ShowDuplicates />,
  },
  {
    id: 'properties',
    label: 'Manage Properties',
    description: 'Add and sync property records',
    icon: Building2,
    group: 'Data',
    render: () => <AddProperties />,
  },
  {
    id: 'contacts',
    label: 'Manage Contacts',
    description: 'Add and sync contact records',
    icon: UserPlus,
    group: 'Data',
    render: () => <AddContacts />,
  },
  {
    id: 'contact-from-invoice',
    label: 'Contact from Invoice',
    description: 'Create a contact from an invoice upload',
    icon: FileUp,
    group: 'Data',
    render: () => <CreateContactFromInvoice />,
  },
  {
    id: 'merge-contacts',
    label: 'Merge Contacts',
    description: 'Combine duplicate contact records',
    icon: GitMerge,
    group: 'Data',
    render: () => <MergeContacts />,
  },
  {
    id: 'timetables',
    label: 'Manage Timetables',
    description: 'Property timetable operations',
    icon: Calendar,
    group: 'Finance',
    render: () => <ManageTimetables />,
  },
  {
    id: 'cashflows',
    label: 'Manage Cashflows',
    description: 'Cashflow import and updates',
    icon: Wallet,
    group: 'Finance',
    render: () => <ManageCashflows />,
  },
  {
    id: 'daily-activity',
    label: 'Daily Activity Jobs',
    description: 'Control daily activity email jobs',
    icon: PlayCircle,
    group: 'Jobs',
    render: () => (
      <JobControl jobId="daily-activity" jobName="Daily Activity Emails" />
    ),
  },
  {
    id: 'expiry-jobs',
    label: 'Expiry Jobs',
    description: 'Schedule and review expiry jobs',
    icon: Timer,
    group: 'Jobs',
    render: () => <ExpiryJobsControl />,
  },
]

const groups = ['Messaging', 'Data', 'Finance', 'Jobs']

function Atmosphere() {
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden="true">
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
  )
}

export default function Utilities() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeId = searchParams.get('tool')
  const activeTool = tools.find((tool) => tool.id === activeId) || null

  const openTool = (id) => setSearchParams({ tool: id })
  const closeTool = () => setSearchParams({})

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
              Kostas&apos; Dashboard
            </p>
            <h1 className="truncate text-base font-semibold text-emerald-950 sm:text-lg">
              {activeTool ? activeTool.label : 'Utilities'}
            </h1>
          </div>
        </div>

        {activeTool ? (
          <button
            type="button"
            onClick={closeTool}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-emerald-900/10 bg-white/80 px-3 py-1.5 text-sm font-medium text-emerald-900 transition hover:border-whatsapp-500/40 hover:text-whatsapp-500"
          >
            <ArrowLeft className="h-3.5 w-3.5" strokeWidth={2.5} />
            All tools
          </button>
        ) : (
          <Link
            to="/"
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-emerald-900/10 bg-white/80 px-3 py-1.5 text-sm font-medium text-emerald-900 transition hover:border-whatsapp-500/40 hover:text-whatsapp-500"
          >
            <ArrowLeft className="h-3.5 w-3.5" strokeWidth={2.5} />
            Home
          </Link>
        )}
      </header>

      <main className="relative z-10 mx-auto flex w-full max-w-5xl flex-1 flex-col px-5 py-10 sm:px-8 sm:py-14">
        {activeTool ? (
          <div className="animate-welcome-rise">
            <div className="mb-8 max-w-xl">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
                {activeTool.group}
              </p>
              <h2 className="font-display text-3xl font-semibold tracking-tight text-emerald-950 sm:text-4xl">
                {activeTool.label}
              </h2>
              <p className="mt-3 text-base text-slate-600">
                {activeTool.description}
              </p>
            </div>

            <div className="rounded-2xl border border-emerald-900/8 bg-white/80 p-4 shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm sm:p-6 [&>*]:mx-auto [&>*]:mt-0 [&>*]:max-w-none [&>*]:rounded-xl [&>*]:border-0 [&>*]:bg-transparent [&>*]:p-0 [&>*]:shadow-none">
              {activeTool.render()}
            </div>
          </div>
        ) : (
          <>
            <div className="mb-10 max-w-xl animate-welcome-rise sm:mb-12">
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
                Utilities
              </p>
              <h2 className="font-display text-4xl font-semibold leading-[1.15] tracking-tight text-emerald-950 sm:text-5xl">
                Pick a tool
              </h2>
              <p className="mt-4 text-base leading-relaxed text-slate-600 sm:text-lg">
                Email, contacts, properties, documents, and scheduled jobs —
                open one at a time.
              </p>
            </div>

            <div className="space-y-10">
              {groups.map((group, groupIndex) => {
                const groupTools = tools.filter((tool) => tool.group === group)
                if (!groupTools.length) return null

                return (
                  <section
                    key={group}
                    className="animate-welcome-rise"
                    style={{ animationDelay: `${60 + groupIndex * 70}ms` }}
                  >
                    <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-800/55">
                      {group}
                    </h3>
                    <nav
                      className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
                      aria-label={`${group} utilities`}
                    >
                      {groupTools.map(
                        ({ id, label, description, icon: Icon }, index) => (
                          <button
                            key={id}
                            type="button"
                            onClick={() => openTool(id)}
                            style={{
                              animationDelay: `${100 + groupIndex * 70 + index * 45}ms`,
                            }}
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
                      )}
                    </nav>
                  </section>
                )
              })}
            </div>
          </>
        )}
      </main>
    </div>
  )
}
