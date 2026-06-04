import React from 'react';
import { 
  UserCheck, 
  Activity, 
  Thermometer, 
  AlertTriangle, 
  Clock, 
  AlertCircle, 
  Cpu, 
  FileText, 
  Shield, 
  CheckCircle,
  Loader2
} from 'lucide-react';

export interface PipelineStage {
  name: string;
  description: string;
  icon: React.ComponentType<any>;
}

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    name: 'Patient Intake',
    description: 'Vitals, demographic registry, and chief complaint text.',
    icon: UserCheck,
  },
  {
    name: 'Symptom Analysis',
    description: 'Extraction of semantic symptom durations and synonym maps.',
    icon: Activity,
  },
  {
    name: 'Differential Diagnosis',
    description: 'Multi-model candidate disease probabilities.',
    icon: Thermometer,
  },
  {
    name: 'Risk Assessment',
    description: 'Long-term risk indexing and comorbid vector calculations.',
    icon: AlertTriangle,
  },
  {
    name: 'Temporal Analysis',
    description: 'Complaints chronology and clinical progression rules.',
    icon: Clock,
  },
  {
    name: 'Emergency Evaluation',
    description: 'Urgency detection and ESI prioritization logic.',
    icon: AlertCircle,
  },
  {
    name: 'Prediction Engine',
    description: 'Voting classifier consensus and ensemble modeling.',
    icon: Cpu,
  },
  {
    name: 'Recommendation Engine',
    description: 'Prescribing guidelines, drug classes, and diagnostic tests.',
    icon: FileText,
  },
  {
    name: 'Supervisor Review',
    description: 'Routing validation and clinical check rules.',
    icon: Shield,
  },
  {
    name: 'Clinical Report',
    description: 'Generated EHR chart, SOAP summaries, and diagnostic notes.',
    icon: CheckCircle,
  },
];

interface PipelineVisualizerProps {
  activeStage?: number; // 0 to 9, or undefined if all completed
  isComplete?: boolean; // if true, force all to completed
}

export default function PipelineVisualizer({ activeStage = 0, isComplete = false }: PipelineVisualizerProps) {
  return (
    <div className="w-full space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {PIPELINE_STAGES.map((stage, idx) => {
          const IconComponent = stage.icon;
          
          let status: 'completed' | 'active' | 'pending' = 'pending';
          if (isComplete) {
            status = 'completed';
          } else if (idx < activeStage) {
            status = 'completed';
          } else if (idx === activeStage) {
            status = 'active';
          }

          return (
            <div 
              key={stage.name} 
              className={`p-4 rounded-xl border transition-all duration-300 flex items-start gap-3.5 bg-card ${
                status === 'completed' 
                  ? 'border-sky-200 bg-sky-50/20' 
                  : status === 'active'
                  ? 'border-sky-600 ring-1 ring-sky-600 bg-sky-50/40 shadow-sm'
                  : 'border-border opacity-65'
              }`}
            >
              <div className={`p-2.5 rounded-lg shrink-0 ${
                status === 'completed'
                  ? 'bg-sky-100 text-sky-850'
                  : status === 'active'
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-100 text-slate-455'
              }`}>
                {status === 'active' && idx !== 9 ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <IconComponent className="w-5 h-5" />
                )}
              </div>
              
              <div className="space-y-0.5 text-left">
                <div className="flex items-center gap-2">
                  <h4 className="font-bold text-xs text-slate-800">
                    {idx + 1}. {stage.name}
                  </h4>
                  {status === 'completed' && (
                    <span className="text-[9px] font-extrabold text-emerald-600 uppercase">Calibrated</span>
                  )}
                  {status === 'active' && (
                    <span className="text-[9px] font-extrabold text-sky-700 uppercase animate-pulse">Running</span>
                  )}
                </div>
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  {stage.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
