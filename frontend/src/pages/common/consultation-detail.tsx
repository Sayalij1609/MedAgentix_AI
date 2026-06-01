import React from 'react';

export default function ConsultationDetail() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Active Consultation</h1>
      <p className="text-muted-foreground">Detailed diagnostic pipeline logs and predictions.</p>
      <div className="h-[400px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [Active Session Details Slot]
      </div>
    </div>
  );
}
