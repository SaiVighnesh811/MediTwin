import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter, MemoryRouter, Routes, Route } from 'react-router-dom';
import Login from '../pages/Login';
import Signup from '../pages/Signup';
import Dashboard from '../pages/Dashboard';
import { AuthProvider } from '../context/AuthContext';
import ProtectedRoute from '../components/ProtectedRoute';

// Mock the API module
vi.mock('../lib/api', () => ({
  authAPI: {
    login: vi.fn(),
    signup: vi.fn(),
    logout: vi.fn(),
    me: vi.fn().mockRejectedValue(new Error('Not authenticated')),
  },
}));

// Mock DigitalTwin component to avoid canvas rendering in jsdom
vi.mock('../components/DigitalTwin', () => ({
  default: () => <div data-testid="digital-twin-mock" />,
}));

import { authAPI } from '../lib/api';

describe('Chapter 11: Auth UI Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('1. shows validation errors on empty login submission', async () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <Login />
        </AuthProvider>
      </MemoryRouter>
    );

    const submitBtn = screen.getByRole('button', { name: /log in/i });
    fireEvent.click(submitBtn);

    expect(await screen.findByText(/email is required/i)).toBeTruthy();
    expect(await screen.findByText(/password is required/i)).toBeTruthy();
    expect(authAPI.login).not.toHaveBeenCalled();
  });

  it('2. validates email format and password length/complexity on signup', async () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <Signup />
        </AuthProvider>
      </MemoryRouter>
    );

    const nameInput = screen.getByLabelText(/full name/i);
    const emailInput = screen.getByLabelText(/email/i);
    const pwInput = screen.getByLabelText(/^password/i);
    const submitBtn = screen.getByRole('button', { name: /create account/i });

    fireEvent.change(nameInput, { target: { value: 'Test User' } });
    fireEvent.change(emailInput, { target: { value: 'not-an-email' } });
    fireEvent.change(pwInput, { target: { value: 'short' } });
    fireEvent.click(submitBtn);

    expect(await screen.findByText(/enter a valid email address/i)).toBeTruthy();
    expect(await screen.findByText(/at least 8 characters/i)).toBeTruthy();
    expect(authAPI.signup).not.toHaveBeenCalled();
  });

  it('3. displays password strength meter on signup page', async () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <Signup />
        </AuthProvider>
      </MemoryRouter>
    );

    const pwInput = screen.getByLabelText(/^password/i);

    // Weak
    fireEvent.change(pwInput, { target: { value: '12345678' } });
    expect(await screen.findByText(/weak/i)).toBeTruthy();

    // Fair (letter + digit + >=8)
    fireEvent.change(pwInput, { target: { value: 'Password123' } });
    expect(await screen.findByText(/fair/i)).toBeTruthy();

    // Strong (>=12 with special char)
    fireEvent.change(pwInput, { target: { value: 'SuperSecret123!' } });
    expect(await screen.findByText(/strong/i)).toBeTruthy();
  });

  it('4. displays server error when login fails', async () => {
    authAPI.login.mockRejectedValueOnce(new Error('Invalid email or password'));

    render(
      <MemoryRouter>
        <AuthProvider>
          <Login />
        </AuthProvider>
      </MemoryRouter>
    );

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'wrong@example.com' } });
    fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'Password123' } });
    fireEvent.click(screen.getByRole('button', { name: /log in/i }));

    expect(await screen.findByText(/invalid email or password/i)).toBeTruthy();
  });

  it('5. redirects unauthenticated user from protected route to /login', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<div>Login Page Mock</div>} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    );

    // Initial auth check will reject, redirecting to /login
    expect(await screen.findByText(/login page mock/i)).toBeTruthy();
  });
});

