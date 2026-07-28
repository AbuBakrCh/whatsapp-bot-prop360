import React from 'react'
import PageShell, { PageIntro, ToolPanel } from '../components/PageShell'
import PropertyCashflowLedger from './PropertyCashflowLedger'

export default function Ledger() {
  return (
    <PageShell title="Ledger" maxWidthClass="max-w-6xl">
      <PageIntro
        eyebrow="Ledger"
        title="Cashflow report"
        description="Review property cashflows, activities, and export ledger data."
      />
      <ToolPanel className="animate-welcome-rise">
        <PropertyCashflowLedger />
      </ToolPanel>
    </PageShell>
  )
}
