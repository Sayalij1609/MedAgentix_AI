import React from 'react';
import { Menu, Bell, Search } from 'lucide-react';
import { useAuth } from '../../context/auth-context';

interface NavbarProps {
  onToggleSidebar: () => void;
  title: string;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar, title }) => {
  const { user } = useAuth();
  const initials = user?.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.charAt(0).toUpperCase() || '?';

  return (
    <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white/90 backdrop-blur-xl px-6 sticky top-0 z-30 w-full shadow-sm">
      <div className="flex items-center gap-4">
        {/* Sidebar toggle */}
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-xl hover:bg-slate-100 text-slate-500 hover:text-slate-800 transition-colors"
          aria-label="Toggle sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Title */}
        <div className="hidden sm:flex items-center gap-2">
          <span className="w-1.5 h-5 rounded-full bg-gradient-to-b from-teal-400 to-sky-500" />
          <h2 className="font-bold text-slate-800 text-base tracking-tight">{title}</h2>
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-2">
        {/* Search (decorative) */}
        <button className="hidden md:flex items-center gap-2 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-400 rounded-xl text-xs font-medium transition-colors">
          <Search className="w-3.5 h-3.5" />
          <span>Search...</span>
        </button>

        {/* Notifications */}
        <button className="p-2 rounded-xl hover:bg-slate-100 text-slate-500 relative transition-colors" aria-label="Notifications">
          <Bell className="w-4.5 h-4.5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-teal-500 rounded-full ring-2 ring-white" />
        </button>

        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-500 to-sky-600 flex items-center justify-center font-bold text-white text-xs shadow-md ring-2 ring-teal-200 ml-1">
          {initials}
        </div>
      </div>
    </header>
  );
};
export default Navbar;
