import React, { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Users, List, Share2 } from 'lucide-react'
import PageShell, { PageIntro, ToolPanel, HubCard } from '../components/PageShell'
import GroupForm from './GroupForm'
import GroupsList from './GroupsList'
import ShareWithGroups from './ShareWithGroups'

export default function Groups() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [editingGroupId, setEditingGroupId] = useState(null)
  const [listRefreshKey, setListRefreshKey] = useState(0)

  const activeId = searchParams.get('tool')
  const openTool = (id) => setSearchParams({ tool: id })
  const closeTool = () => {
    setEditingGroupId(null)
    setSearchParams({})
  }

  const handleSaved = () => {
    setEditingGroupId(null)
    setListRefreshKey((k) => k + 1)
    openTool('list')
  }

  const tools = {
    form: {
      id: 'form',
      label: editingGroupId ? 'Edit Group' : 'Create Group',
      description: editingGroupId
        ? 'Update an existing contact group'
        : 'Create a new contact group',
      icon: Users,
    },
    list: {
      id: 'list',
      label: 'Groups List',
      description: 'Browse and edit your contact groups',
      icon: List,
    },
    share: {
      id: 'share',
      label: 'Share with Groups',
      description: 'Share content with selected groups',
      icon: Share2,
    },
  }

  const activeTool = tools[activeId] || null

  return (
    <PageShell
      title={activeTool ? activeTool.label : 'Groups'}
      onBack={activeTool ? closeTool : undefined}
      backLabel={activeTool ? 'All tools' : 'Home'}
    >
      {activeTool ? (
        <div className="animate-welcome-rise">
          <div className="mb-8 max-w-xl">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-whatsapp-500">
              Groups
            </p>
            <h2 className="font-display text-3xl font-semibold tracking-tight text-emerald-950 sm:text-4xl">
              {activeTool.label}
            </h2>
            <p className="mt-3 text-base text-slate-600">
              {activeTool.description}
            </p>
          </div>
          <ToolPanel>
            {activeId === 'form' && (
              <GroupForm
                editingGroupId={editingGroupId}
                onCancel={() => {
                  setEditingGroupId(null)
                  if (editingGroupId) openTool('list')
                  else closeTool()
                }}
                onSaved={handleSaved}
              />
            )}
            {activeId === 'list' && (
              <GroupsList
                refreshKey={listRefreshKey}
                onEdit={(groupId) => {
                  setEditingGroupId(groupId)
                  openTool('form')
                }}
              />
            )}
            {activeId === 'share' && <ShareWithGroups />}
          </ToolPanel>
        </div>
      ) : (
        <>
          <PageIntro
            eyebrow="Groups"
            title="Contact groups"
            description="Create groups, manage members, and share with them — open one at a time."
          />
          <nav
            className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
            aria-label="Groups tools"
          >
            {Object.values(tools).map((tool, index) => (
              <HubCard
                key={tool.id}
                label={tool.id === 'form' ? 'Create Group' : tool.label}
                description={
                  tool.id === 'form'
                    ? 'Create a new contact group'
                    : tool.description
                }
                icon={tool.icon}
                onClick={() => {
                  setEditingGroupId(null)
                  openTool(tool.id)
                }}
                delayMs={80 + index * 55}
              />
            ))}
          </nav>
        </>
      )}
    </PageShell>
  )
}
