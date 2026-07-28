import React from 'react'

/** Soft green backdrop used across dashboard pages. */
export default function Atmosphere() {
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
