import React from 'react';

export default function PatientDashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Patient Workspace</h1>
      <p className="text-muted-foreground">Overview of your past predictions, active cases, and treatments.</p>
      <div className="h-[400px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [Patient Summary Grid Slot]
      </div>
    </div>
  );
}
