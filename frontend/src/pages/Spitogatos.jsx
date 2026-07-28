import React from 'react'
import { useSearchParams } from 'react-router-dom'
import { Filter, Search, Share2, Bookmark } from 'lucide-react'
import PageShell, { PageIntro, ToolPanel, HubCard } from '../components/PageShell'
import SpitogatosCrawler from './SpitogatosCrawler'
import SpitogatosFilters from './SpitogatosFilters'
import SharePropertyJobControl from './SharePropertyJobControl'
import PropertyFiltersList from './PropertyFiltersList'

const tools = [
  {
    id: 'filters',
    label: 'Store Client Filters',
    description: 'Save client search filters for Spitogatos',
    icon: Filter,
    render: () => <SpitogatosFilters />,
  },
  {
    id: 'crawler',
    label: 'Spitogatos Crawler',
    description: 'Run and monitor listing crawls',
    icon: Search,
    render: () => <SpitogatosCrawler />,
  },
  {
    id: 'share-job',
    label: 'Share Property Job',
    description: 'Control property sharing jobs',
    icon: Share2,
    render: () => <SharePropertyJobControl />,
  },
  {
    id: 'saved-filters',
    label: 'Saved Filters',
    description: 'Browse and manage stored filters',
    icon: Bookmark,
    render: () => <PropertyFiltersList />,
  },
]

export default function Spitogatos() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeId = searchParams.get('tool')
  const activeTool = tools.find((tool) => tool.id === activeId) || null

  const openTool = (id) => setSearchParams({ tool: id })
  const closeTool = () => setSearchParams({})

  return (
    <PageShell
      title={activeTool ? activeTool.label : 'Spitogatos'}
      onBack={activeTool ? closeTool : undefined}
      backLabel={activeTool ? 'All tools' : 'Home'}
    >
      {activeTool ? (
        <div className="animate-welcome-rise">
          <div className="mb-8 max-w-xl">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
              Spitogatos
            </p>
            <h2 className="font-display text-3xl font-semibold tracking-tight text-emerald-950 sm:text-4xl">
              {activeTool.label}
            </h2>
            <p className="mt-3 text-base text-slate-600">
              {activeTool.description}
            </p>
          </div>
          <ToolPanel>{activeTool.render()}</ToolPanel>
        </div>
      ) : (
        <>
          <PageIntro
            eyebrow="Spitogatos"
            title="Property listings"
            description="Filters, crawler, sharing jobs, and saved searches — open one at a time."
          />
          <nav
            className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
            aria-label="Spitogatos tools"
          >
            {tools.map((tool, index) => (
              <HubCard
                key={tool.id}
                label={tool.label}
                description={tool.description}
                icon={tool.icon}
                onClick={() => openTool(tool.id)}
                delayMs={80 + index * 55}
              />
            ))}
          </nav>
        </>
      )}
    </PageShell>
  )
}
