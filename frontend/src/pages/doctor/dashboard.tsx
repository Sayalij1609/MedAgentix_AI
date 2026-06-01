import React from 'react';

export default function DoctorDashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Physician Dashboard</h1>
      <p className="text-muted-foreground">Roster overview and unassigned consultation triage lists.</p>
      <div className="h-[400px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [Doctor Operations Grid Slot]
      </div>
    </div>
  );
}
