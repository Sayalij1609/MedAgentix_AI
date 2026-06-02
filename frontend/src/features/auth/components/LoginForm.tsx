import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, Eye, EyeOff, Loader2, AlertCircle } from 'lucide-react';
import { useAuth } from '../../../context/auth-context';
import { AuthService } from '../services/auth-service';
import { LoginPayload, FormErrors } from '../types';
import { ROUTES } from '../../../routes/config';

export const LoginForm: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();

  // Form State
  const [formData, setFormData] = useState<LoginPayload>({
    email: '',
    password: '',
  });

  // Password Visibility Toggle
  const [showPassword, setShowPassword] = useState(false);

  // Status & Validation States
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<FormErrors<LoginPayload>>({});
  const [apiError, setApiError] = useState<string | null>(null);

  // Field change handler
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    
    // Clear field-specific error as user types
    if (errors[name as keyof LoginPayload]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
    if (apiError) setApiError(null);
  };

  // Basic Form Validation
  const validateForm = (): boolean => {
    const newErrors: FormErrors<LoginPayload> = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.email.trim()) {
      newErrors.email = 'Clinical email address is required';
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid clinical email (e.g. name@medagentix.ai)';
    }

    if (!formData.password) {
      newErrors.password = 'Security password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Submit Handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);

    if (!validateForm()) return;

    setIsLoading(true);
    try {
      // Simulate API verification
      const response = await AuthService.login(formData);
      
      // Update global context session (auto-triggers layout redirect)
      login(response.access_token, response.user);
      
      // Safety redirect fallback
      const targetRoute = response.user.role === 'doctor' 
        ? ROUTES.DOCTOR_DASHBOARD 
        : ROUTES.PATIENT_DASHBOARD;
      navigate(targetRoute);
    } catch (err: any) {
      setApiError(err.message || 'An unexpected authentication error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold tracking-tight text-foreground font-display bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          Clinical Portal Sign In
        </h2>
        <p className="text-muted-foreground text-sm">
          Enter your authorized clinical credentials to synchronize your medical console.
        </p>
      </div>

      {/* API Error Display */}
      <AnimatePresence mode="wait">
        {apiError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-start gap-3 p-4 bg-danger/10 border border-danger/25 text-danger rounded-xl text-xs leading-relaxed"
          >
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-danger">Credential verification failed</span>
              <p className="opacity-90">{apiError}</p>
              <div className="pt-2 text-[10px] text-slate-500 font-mono">
                Tip: Avoid typing 'error@medagentix.ai' which triggers mock errors.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Email Input */}
        <div className="space-y-1.5">
          <label htmlFor="email" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            Clinical Email Address
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Mail className="w-4 h-4" />
            </div>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              value={formData.email}
              onChange={handleChange}
              disabled={isLoading}
              className={`w-full pl-10 pr-4 py-2.5 bg-card border rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-secondary/20 focus:border-secondary ${
                errors.email 
                  ? 'border-danger focus:ring-danger/20 focus:border-danger' 
                  : 'border-border'
              }`}
              placeholder="e.g. doctor@medagentix.ai"
            />
          </div>
          {errors.email && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-xs font-medium text-danger flex items-center gap-1.5"
            >
              <span className="w-1 h-1 rounded-full bg-danger"></span>
              {errors.email}
            </motion.p>
          )}
        </div>

        {/* Password Input */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <label htmlFor="password" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Security Password
            </label>
            <a 
              href="#" 
              onClick={(e) => { e.preventDefault(); alert("Mock recovery code dispatched to clinical administrator."); }}
              className="text-xs text-secondary hover:underline font-medium"
            >
              Reset credentials?
            </a>
          </div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Lock className="w-4 h-4" />
            </div>
            <input
              id="password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              value={formData.password}
              onChange={handleChange}
              disabled={isLoading}
              className={`w-full pl-10 pr-10 py-2.5 bg-card border rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-secondary/20 focus:border-secondary ${
                errors.password 
                  ? 'border-danger focus:ring-danger/20 focus:border-danger' 
                  : 'border-border'
              }`}
              placeholder="••••••••"
            />
            <button
              type="button"
              onClick={() => setShowPassword((prev) => !prev)}
              tabIndex={-1}
              className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none"
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
          {errors.password && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-xs font-medium text-danger flex items-center gap-1.5"
            >
              <span className="w-1 h-1 rounded-full bg-danger"></span>
              {errors.password}
            </motion.p>
          )}
        </div>

        {/* Demo Hint Helper */}
        <div className="p-3 bg-secondary/5 rounded-xl border border-secondary/15 space-y-1">
          <p className="text-[11px] font-semibold text-secondary-foreground flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-secondary animate-pulse" />
            Developer Simulation Controls:
          </p>
          <div className="text-[10px] text-slate-500 leading-tight space-y-0.5">
            <div>• Type <span className="font-mono bg-slate-100 dark:bg-slate-800 px-1 py-0.5 rounded text-secondary">doctor@medagentix.ai</span> for Doctor view.</div>
            <div>• Type <span className="font-mono bg-slate-100 dark:bg-slate-800 px-1 py-0.5 rounded text-secondary">patient@medagentix.ai</span> for Patient view.</div>
          </div>
        </div>

        {/* Submit Action */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-primary text-white font-semibold rounded-xl text-sm hover:bg-primary/95 focus:ring-4 focus:ring-primary/20 transition-all shadow-md active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Verifying clinical credentials...</span>
            </>
          ) : (
            <span>Synchronize Console</span>
          )}
        </button>
      </form>

      {/* Footer Registration Redirect */}
      <div className="text-center pt-2">
        <p className="text-xs text-muted-foreground">
          New clinical agent or patient registration?{' '}
          <Link to={ROUTES.REGISTER} className="text-secondary font-semibold hover:underline">
            Request Authorized Account
          </Link>
        </p>
      </div>
    </div>
  );
};
export default LoginForm;
