import React from 'react'
import { ExternalLink } from 'lucide-react'
import PageShell, { PageIntro } from '../components/PageShell'

const links = [
  {
    group: 'OpenAI',
    items: [
      {
        label: 'OpenAI Platform Dashboard',
        description: 'Organization billing and usage overview',
        href: 'https://platform.openai.com/settings/organization/billing/overview',
      },
    ],
  },
]

export default function ImportantLinks() {
  return (
    <PageShell title="Important links">
      <PageIntro
        eyebrow="Important links"
        title="Quick access"
        description="External tools and dashboards you reach often."
      />

      <div className="space-y-10">
        {links.map((section, sectionIndex) => (
          <section
            key={section.group}
            className="animate-welcome-rise"
            style={{ animationDelay: `${60 + sectionIndex * 70}ms` }}
          >
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-800/55">
              {section.group}
            </h3>
            <nav className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {section.items.map((item, index) => (
                <a
                  key={item.href}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ animationDelay: `${100 + index * 45}ms` }}
                  className="group relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-emerald-900/8 bg-white/80 p-5 shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm transition duration-300 animate-welcome-rise hover:-translate-y-0.5 hover:border-whatsapp-500/35 hover:bg-white hover:shadow-[0_12px_32px_-12px_rgba(45,188,58,0.35)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-whatsapp-500"
                >
                  <div className="flex items-start justify-between gap-3">
                    <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-whatsapp-100 to-emerald-50 text-whatsapp-500 ring-1 ring-emerald-900/5 transition duration-300 group-hover:from-whatsapp-500 group-hover:to-emerald-600 group-hover:text-white group-hover:ring-transparent">
                      <ExternalLink className="h-5 w-5" strokeWidth={2} />
                    </span>
                    <ExternalLink className="h-4 w-4 text-emerald-900/20 transition duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-whatsapp-500" />
                  </div>
                  <div>
                    <div className="text-base font-semibold text-emerald-950 transition group-hover:text-whatsapp-500">
                      {item.label}
                    </div>
                    <div className="mt-1.5 text-sm leading-snug text-slate-500">
                      {item.description}
                    </div>
                  </div>
                </a>
              ))}
            </nav>
          </section>
        ))}
      </div>
    </PageShell>
  )
}
