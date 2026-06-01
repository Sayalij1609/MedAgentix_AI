import React from 'react';

export default function DoctorQueue() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Patient Triage Queue</h1>
      <p className="text-muted-foreground"> Roster grid containing unassigned consultation requests.</p>
      <div className="h-[450px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [Triage Queue Data Table Slot]
      </div>
    </div>
  );
}
