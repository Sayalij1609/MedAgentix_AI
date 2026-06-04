import React from 'react';
import { Menu, Bell } from 'lucide-react';

interface NavbarProps {
  onToggleSidebar: () => void;
  title: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  onToggleSidebar,
  title,
}) => {
  return (
    <header className="flex h-16 items-center justify-between border-b border-border bg-card px-6 sticky top-0 z-30 w-full text-left">
      <div className="flex items-center gap-4">
        {/* Toggle Burger */}
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Toggle sidebar menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <h2 className="font-semibold text-md text-foreground hidden sm:block">
          {title}
        </h2>
      </div>

      <div className="flex items-center gap-4">
        {/* Alerts Center Feed */}
        <button 
          className="p-2 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground relative transition-colors"
          aria-label="View notifications"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full ring-2 ring-card"></span>
        </button>
      </div>
    </header>
  );
};
export default Navbar;
