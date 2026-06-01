import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Mail, Lock, Eye, EyeOff, Loader2, AlertCircle, Activity, Stethoscope } from 'lucide-react';
import { useAuth, UserRole } from '../../../context/auth-context';
import { AuthService } from '../services/auth-service';
import { RegisterPayload, FormErrors } from '../types';
import { ROUTES } from '../../../routes/config';

export const RegisterForm: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();

  // Form State
  const [formData, setFormData] = useState<RegisterPayload>({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'Patient', // Default role selection
  });

  // Password Visibility Toggle
  const [showPassword, setShowPassword] = useState(false);

  // Status & Validation States
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<FormErrors<RegisterPayload>>({});
  const [apiError, setApiError] = useState<string | null>(null);

  // Field change handler
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    
    // Clear field-specific error as user types
    if (errors[name as keyof RegisterPayload]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
    if (apiError) setApiError(null);
  };

  // Role click selector handler
  const handleRoleSelect = (role: UserRole) => {
    setFormData((prev) => ({ ...prev, role }));
    if (errors.role) {
      setErrors((prev) => ({ ...prev, role: undefined }));
    }
  };

  // Basic Form Validation
  const validateForm = (): boolean => {
    const newErrors: FormErrors<RegisterPayload> = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.name.trim()) {
      newErrors.name = 'Full identity name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Clinical email address is required';
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid clinical email (e.g. name@medagentix.ai)';
    }

    if (!formData.password) {
      newErrors.password = 'Security password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    if (!formData.role) {
      newErrors.role = 'Please specify your platform role';
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
      // Simulate API registration
      const response = await AuthService.register(formData);
      
      // Auto-login after registration
      login(response.access_token, response.user);
      
      // Safety redirect fallback
      const targetRoute = response.user.role === 'Doctor' 
        ? ROUTES.DOCTOR_DASHBOARD 
        : ROUTES.PATIENT_DASHBOARD;
      navigate(targetRoute);
    } catch (err: any) {
      setApiError(err.message || 'An unexpected registration error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold tracking-tight text-foreground font-display bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          Clinical Registration
        </h2>
        <p className="text-muted-foreground text-sm">
          Initialize your authorized medical practitioner or patient record dashboard.
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
              <span className="font-semibold text-danger">Registration attempt failed</span>
              <p className="opacity-90">{apiError}</p>
              <div className="pt-2 text-[10px] text-slate-500 font-mono">
                Tip: Avoid registering with 'taken@medagentix.ai' which triggers mock duplicates.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Visual Role Selector */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">
            Select Platform Access Role
          </label>
          <div className="grid grid-cols-2 gap-3">
            {/* Patient Option */}
            <button
              type="button"
              onClick={() => handleRoleSelect('Patient')}
              disabled={isLoading}
              className={`flex flex-col items-center gap-2 p-3 text-center border-2 rounded-xl transition-all ${
                formData.role === 'Patient'
                  ? 'border-secondary bg-secondary/5 shadow-glow text-secondary-foreground'
                  : 'border-border bg-card hover:border-slate-300 text-muted-foreground'
              }`}
            >
              <div className={`p-2 rounded-lg ${formData.role === 'Patient' ? 'bg-secondary/15' : 'bg-slate-100'}`}>
                <Activity className={`w-5 h-5 ${formData.role === 'Patient' ? 'text-secondary' : 'text-slate-500'}`} />
              </div>
              <div>
                <p className="text-xs font-bold">Patient Portal</p>
                <p className="text-[10px] opacity-75 mt-0.5 line-clamp-1">Assess active symptoms</p>
              </div>
            </button>

            {/* Doctor Option */}
            <button
              type="button"
              onClick={() => handleRoleSelect('Doctor')}
              disabled={isLoading}
              className={`flex flex-col items-center gap-2 p-3 text-center border-2 rounded-xl transition-all ${
                formData.role === 'Doctor'
                  ? 'border-primary bg-primary/5 shadow-md text-primary'
                  : 'border-border bg-card hover:border-slate-300 text-muted-foreground'
              }`}
            >
              <div className={`p-2 rounded-lg ${formData.role === 'Doctor' ? 'bg-primary/10' : 'bg-slate-100'}`}>
                <Stethoscope className={`w-5 h-5 ${formData.role === 'Doctor' ? 'text-primary' : 'text-slate-500'}`} />
              </div>
              <div>
                <p className="text-xs font-bold">Clinical Doctor</p>
                <p className="text-[10px] opacity-75 mt-0.5 line-clamp-1">Triage patient cases</p>
              </div>
            </button>
          </div>
          {errors.role && (
            <p className="text-xs font-medium text-danger">{errors.role}</p>
          )}
        </div>

        {/* Identity Name Input */}
        <div className="space-y-1.5">
          <label htmlFor="name" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            Full Identity Name
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <User className="w-4 h-4" />
            </div>
            <input
              id="name"
              name="name"
              type="text"
              autoComplete="name"
              value={formData.name}
              onChange={handleChange}
              disabled={isLoading}
              className={`w-full pl-10 pr-4 py-2.5 bg-card border rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-secondary/20 focus:border-secondary ${
                errors.name 
                  ? 'border-danger focus:ring-danger/20 focus:border-danger' 
                  : 'border-border'
              }`}
              placeholder="e.g. Dr. Arthur Pendelton or Patient Jenkins"
            />
          </div>
          {errors.name && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-xs font-medium text-danger flex items-center gap-1.5"
            >
              <span className="w-1 h-1 rounded-full bg-danger"></span>
              {errors.name}
            </motion.p>
          )}
        </div>

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
              placeholder="e.g. resident@medagentix.ai"
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

        {/* Password Inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {/* Password */}
          <div className="space-y-1.5">
            <label htmlFor="password" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="new-password"
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
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {errors.password && (
              <p className="text-xs font-medium text-danger">{errors.password}</p>
            )}
          </div>

          {/* Confirm Password */}
          <div className="space-y-1.5">
            <label htmlFor="confirmPassword" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Confirm Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type={showPassword ? 'text' : 'password'}
                autoComplete="new-password"
                value={formData.confirmPassword}
                onChange={handleChange}
                disabled={isLoading}
                className={`w-full pl-10 pr-4 py-2.5 bg-card border rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-secondary/20 focus:border-secondary ${
                  errors.confirmPassword 
                    ? 'border-danger focus:ring-danger/20 focus:border-danger' 
                    : 'border-border'
                }`}
                placeholder="••••••••"
              />
            </div>
            {errors.confirmPassword && (
              <p className="text-xs font-medium text-danger">{errors.confirmPassword}</p>
            )}
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
              <span>Instantiating secure console...</span>
            </>
          ) : (
            <span>Request Platform Registration</span>
          )}
        </button>
      </form>

      {/* Footer Login Redirect */}
      <div className="text-center pt-2">
        <p className="text-xs text-muted-foreground">
          Already registered?{' '}
          <Link to={ROUTES.LOGIN} className="text-secondary font-semibold hover:underline">
            Clinical Portal Login
          </Link>
        </p>
      </div>
    </div>
  );
};
export default RegisterForm;
