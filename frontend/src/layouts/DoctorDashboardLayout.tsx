import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sidebar, SidebarNavItem } from '../components/common/Sidebar';
import { Navbar } from '../components/common/Navbar';
import { ROUTES } from '../routes/config';
import { Activity, ClipboardList, Users } from 'lucide-react';

const DOCTOR_NAV_ITEMS: SidebarNavItem[] = [
  {
    name: 'Clinic Overview',
    path: ROUTES.DOCTOR_DASHBOARD,
    icon: Activity,
  },
  {
    name: 'Triage Queue',
    path: ROUTES.DOCTOR_QUEUE,
    icon: ClipboardList,
  },
  {
    name: 'Emergency Cases',
    path: ROUTES.DOCTOR_TRIAGE,
    icon: Users,
  },
];

export const DoctorDashboardLayout: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleToggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const handleCloseSidebar = () => {
    setIsSidebarOpen(false);
  };

  return (
    <div className="min-h-screen flex bg-background text-foreground transition-colors duration-300">
      {/* Mobile Drawer Overlay */}
      {isSidebarOpen && (
        <div
          onClick={handleCloseSidebar}
          className="fixed inset-0 z-35 bg-black/50 backdrop-blur-sm md:hidden"
        />
      )}

      {/* Sidebar Component */}
      <Sidebar
        navItems={DOCTOR_NAV_ITEMS}
        isOpen={isSidebarOpen}
        onClose={handleCloseSidebar}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto">
        <Navbar
          onToggleSidebar={handleToggleSidebar}
          title="Clinical Triage Control"
        />

        {/* View Page Outlet */}
        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  );
};
export default DoctorDashboardLayout;
