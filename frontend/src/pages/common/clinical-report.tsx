import React from 'react';

export default function ClinicalReport() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Clinical Report Card</h1>
      <p className="text-muted-foreground">Downloadable case summaries and verified prescriptions.</p>
      <div className="h-[450px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [PDF Printable Summary Chart Slot]
      </div>
    </div>
  );
}
