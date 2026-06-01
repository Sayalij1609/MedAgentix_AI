import React from 'react';

export default function PatientIntake() {
  return (
    <div className="space-y-6 h-full flex flex-col">
      <h1 className="text-3xl font-bold tracking-tight">Symptom Assessment</h1>
      <p className="text-muted-foreground">Describe your symptoms to interact with the multi-agent diagnostic engine.</p>
      <div className="flex-1 min-h-[450px] border border-border rounded-xl flex items-center justify-center text-muted-foreground bg-card/20">
        [Chat Intake Console Slot]
      </div>
    </div>
  );
}
