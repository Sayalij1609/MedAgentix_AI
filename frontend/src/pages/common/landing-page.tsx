import React from 'react';

export default function LandingPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center p-6">
      <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent animate-fade-in">
        MedAgentix AI
      </h1>
      <p className="mt-4 text-muted-foreground max-w-xl">
        Intelligent Multi-Agent Clinical Diagnostics & Retrievable Knowledge Base.
      </p>
      <div className="mt-8 flex gap-4">
        <a href="/login" className="px-6 py-3 bg-primary text-primary-foreground font-semibold rounded-lg hover:opacity-90 transition shadow-lg">
          Get Started
        </a>
      </div>
    </div>
  );
}
