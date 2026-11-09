import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import GuestRoute from './components/GuestRoute';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import DashboardOverview from './pages/DashboardOverview';
import Wizard from './pages/Wizard';
import AnalysisHistory from './pages/AnalysisHistory';
import AnalysisDetail from './pages/AnalysisDetail';

const pageVariants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
};

function PageWrapper({ children }) {
  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {children}
    </motion.div>
  );
}

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
        <Route
          path="/login"
          element={
            <PageWrapper>
              <GuestRoute><Login /></GuestRoute>
            </PageWrapper>
          }
        />
        <Route
          path="/signup"
          element={
            <PageWrapper>
              <GuestRoute><Signup /></GuestRoute>
            </PageWrapper>
          }
        />
        <Route
          path="/dashboard"
          element={
            <PageWrapper>
              <ProtectedRoute><DashboardOverview /></ProtectedRoute>
            </PageWrapper>
          }
        />
        <Route
          path="/dashboard/new"
          element={
            <PageWrapper>
              <ProtectedRoute><Wizard /></ProtectedRoute>
            </PageWrapper>
          }
        />
        <Route
          path="/dashboard/history"
          element={
            <PageWrapper>
              <ProtectedRoute><AnalysisHistory /></ProtectedRoute>
            </PageWrapper>
          }
        />
        <Route
          path="/dashboard/analysis/:id"
          element={
            <PageWrapper>
              <ProtectedRoute><AnalysisDetail /></ProtectedRoute>
            </PageWrapper>
          }
        />
      </Routes>
    </AnimatePresence>
  );
}

// Ambient pointer glow (desktop hover devices only)
function AmbientGlow() {
  useEffect(() => {
    const mq = window.matchMedia('(hover: hover) and (pointer: fine)');
    if (!mq.matches) return;

    const glow = document.querySelector('.ambient-glow');
    if (!glow) return;

    let ticking = false;
    function onMove(e) {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        glow.style.left = e.clientX + 'px';
        glow.style.top = e.clientY + 'px';
        ticking = false;
      });
    }

    document.addEventListener('mousemove', onMove, { passive: true });
    return () => document.removeEventListener('mousemove', onMove);
  }, []);

  return <div className="ambient-glow" aria-hidden="true" />;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AmbientGlow />
        <AnimatedRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
