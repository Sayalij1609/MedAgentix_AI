import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { LogOut, ChevronRight } from 'lucide-react';
import { useAuth } from '../../context/auth-context';

export interface SidebarNavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface SidebarProps {
  navItems: SidebarNavItem[];
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ navItems, isOpen, onClose }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const initials = user?.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.charAt(0).toUpperCase() || '?';

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 w-64 flex flex-col transition-transform duration-300 md:translate-x-0 md:static shrink-0 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}
      style={{ background: 'linear-gradient(160deg, #0f172a 0%, #0d2e4a 50%, #0a3d62 100%)' }}
    >
      {/* Grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(56,189,248,0.07) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(56,189,248,0.07) 1px, transparent 1px)
          `,
          backgroundSize: '28px 28px',
        }}
      />

      {/* Ambient glow */}
      <div className="absolute -top-20 -left-10 w-56 h-56 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 right-0 w-40 h-40 bg-sky-500/8 rounded-full blur-3xl pointer-events-none" />

      {/* Brand Header */}
      <div className="relative z-10 flex h-16 items-center px-5 border-b border-white/10 justify-between shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500/40 to-sky-500/30 border border-teal-400/40 flex items-center justify-center font-black text-sm text-teal-200">
            M
          </div>
          <div>
            <p className="font-extrabold text-white text-sm leading-none">MedAgentix</p>
            <p className="text-teal-400 text-[9px] font-semibold mt-0.5 tracking-wide">Clinical Portal</p>
          </div>
        </div>
        {/* Mobile close */}
        <button onClick={onClose} className="md:hidden p-1.5 rounded-lg hover:bg-white/10 text-white/60" aria-label="Close sidebar">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Nav Menu */}
      <nav className="relative z-10 flex-1 px-3 py-5 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <button
              key={item.name}
              onClick={() => { navigate(item.path); onClose(); }}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 group ${
                isActive
                  ? 'bg-gradient-to-r from-teal-500/25 to-sky-500/15 text-white border border-teal-400/30 shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-white/8 border border-transparent'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-teal-300' : 'text-slate-500 group-hover:text-teal-400'}`} />
              <span className="flex-1 text-left">{item.name}</span>
              {isActive && <ChevronRight className="w-3 h-3 text-teal-400" />}
            </button>
          );
        })}
      </nav>

      {/* User Profile + Logout */}
      <div className="relative z-10 p-4 border-t border-white/10 space-y-3 shrink-0">
        {user && (
          <div className="flex items-center gap-3 px-2 py-1">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-teal-500 to-sky-600 flex items-center justify-center font-bold text-white text-sm shadow-md ring-2 ring-teal-500/30 shrink-0">
              {initials}
            </div>
            <div className="truncate min-w-0">
              <p className="text-xs font-bold text-white truncate">{user.name || 'User'}</p>
              <p className="text-[10px] text-slate-400 truncate">{user.email}</p>
              <span className="text-[9px] uppercase font-extrabold text-teal-400 tracking-wider">{user.role}</span>
            </div>
          </div>
        )}
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold text-rose-400 hover:bg-rose-500/10 hover:text-rose-300 border border-transparent hover:border-rose-500/20 transition-all duration-200"
        >
          <LogOut className="w-4 h-4 shrink-0" />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
};
export default Sidebar;
