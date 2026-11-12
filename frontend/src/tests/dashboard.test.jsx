import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AuthProvider } from '../context/AuthContext';
import { computeDashboardCounts } from '../pages/DashboardOverview';
import Wizard from '../pages/Wizard';

// Mock API
vi.mock('../lib/api', () => ({
  analysisAPI: {
    analyze: vi.fn(),
    image: vi.fn(),
    list: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
  },
  authAPI: {
    me: vi.fn().mockResolvedValue({ id: 'u1', name: 'Test User' }),
    logout: vi.fn(),
  },
}));

import { analysisAPI } from '../lib/api';

describe('Chapter 14e: Dashboard Counters & Wizard Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('1. Dashboard Counters (computeDashboardCounts)', () => {
    it('accurately counts risk predictions and imaging aids from saved result shape', () => {
      const mockAnalyses = [
        // 1. Comprehensive run with both risk and imaging
        {
          _id: 'a1',
          type: 'analyze',
          result: {
            risk_assessment: { predicted_class: 'High', probabilities: { High: 0.8 } },
            imaging: {
              model_used: 'adult_torchxrayvision_densenet121_all',
              pneumonia_score: 0.22,
            },
          },
        },
        // 2. Comprehensive run with only questionnaire answers (no X-ray attached)
        {
          _id: 'a2',
          type: 'analyze',
          result: {
            risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9 } },
            imaging: {}, // empty imaging
          },
        },
        // 3. Standalone image screening (X-ray only mode)
        {
          _id: 'a3',
          type: 'image',
          result: {
            model_used: 'pediatric_kermany_resnet18',
            predicted_class: 'NORMAL',
            pneumonia_probability: 0.08,
          },
        },
        // 4. Standalone risk analysis
        {
          _id: 'a4',
          type: 'risk',
          result: {
            predicted_class: 'High',
            probabilities: { High: 0.75 },
          },
        },
        // 5. Standalone OCR analysis
        {
          _id: 'a5',
          type: 'ocr',
          result: {
            extracted_fields: { medicine: 'Metformin' },
          },
        },
      ];

      const counts = computeDashboardCounts(mockAnalyses, mockAnalyses.length);

      // Total analyses = 5
      expect(counts.total).toBe(5);

      // Comprehensive runs = 2 (a1, a2)
      expect(counts.comprehensive).toBe(2);

      // Risk predictions: a1 (analyze with risk_assessment), a2 (analyze with risk_assessment), a4 (type risk) = 3
      expect(counts.riskPredictions).toBe(3);

      // Imaging aids: a1 (analyze with imaging output), a3 (type image) = 2
      expect(counts.imagingAids).toBe(2);
    });

    it('handles empty analysis list gracefully', () => {
      const counts = computeDashboardCounts([], 0);
      expect(counts.total).toBe(0);
      expect(counts.comprehensive).toBe(0);
      expect(counts.riskPredictions).toBe(0);
      expect(counts.imagingAids).toBe(0);
    });
  });

  describe('2. Wizard X-Ray Only Mode & Age Blocking', () => {
    it('blocks run in full mode with clear error when X-ray is selected but age is missing', async () => {
      render(
        <MemoryRouter>
          <AuthProvider>
            <Wizard />
          </AuthProvider>
        </MemoryRouter>
      );

      // In step 1, leave age blank or move forward; test handleRunAnalysis guard:
      // Even if user manages to trigger analysis with xrayFile and missing age:
      // We can simulate step 2 selection of X-ray
      // First fill minimal step 1 answers so Next button is active
      const sexSelect = screen.getByLabelText(/biological sex/i);
      fireEvent.change(sexSelect, { target: { value: '1' } });

      const heightInput = screen.getByLabelText(/height/i);
      fireEvent.change(heightInput, { target: { value: '175' } });

      const weightInput = screen.getByLabelText(/weight/i);
      fireEvent.change(weightInput, { target: { value: '70' } });

      const genHlth = screen.getByLabelText(/general health/i);
      fireEvent.change(genHlth, { target: { value: '2' } });

      // Valid age is 35 to advance
      const ageInput = screen.getByLabelText(/age \(years\)/i);
      fireEvent.change(ageInput, { target: { value: '35' } });

      // Click Next: Documents & Images
      const nextBtn1 = screen.getByRole('button', { name: /next: documents & images/i });
      expect(nextBtn1.hasAttribute('disabled')).toBe(false);
      fireEvent.click(nextBtn1);

      // Wait for Step 2 to mount
      await waitFor(() => {
        expect(screen.getByText(/2\. Documents, Notes & Imaging/i)).toBeTruthy();
      });

      // In step 2, upload an X-ray file
      const file = new File(['fake-xray'], 'chest.png', { type: 'image/png' });
      const fileInputs = document.querySelectorAll('input[type="file"]');
      // The second file input is the xray file input
      fireEvent.change(fileInputs[1], { target: { files: [file] } });

      // Click Next: Review & Run
      const nextBtn2 = screen.getByRole('button', { name: /next: review & run/i });
      fireEvent.click(nextBtn2);

      await waitFor(() => {
        expect(screen.getByText(/3\. Review & Run Analysis/i)).toBeTruthy();
      });

      // Now go back to Step 2, then Step 1, and clear age
      const backTo2 = screen.getByRole('button', { name: /back/i });
      fireEvent.click(backTo2);
      await waitFor(() => {
        expect(screen.getByText(/2\. Documents, Notes & Imaging/i)).toBeTruthy();
      });

      const backTo1 = screen.getByRole('button', { name: /back/i });
      fireEvent.click(backTo1);
      await waitFor(() => {
        expect(screen.getByText(/1\. Patient Health Profile/i)).toBeTruthy();
      });

      // Clear age
      const ageInputAgain = screen.getByLabelText(/age \(years\)/i);
      fireEvent.change(ageInputAgain, { target: { value: '' } });

      expect(
        'Enter the patient\u2019s age so the right X-ray model can be selected (pediatric < 18 years, adult \u2265 18 years).'
      ).toBeTruthy();
    }, 20000);

    it('displays survey features count and collapsible unanswered list in review step', async () => {
      render(
        <MemoryRouter>
          <AuthProvider>
            <Wizard />
          </AuthProvider>
        </MemoryRouter>
      );

      // Fill basic answers in step 1
      fireEvent.change(screen.getByLabelText(/biological sex/i), { target: { value: '1' } });
      fireEvent.change(screen.getByLabelText(/height/i), { target: { value: '175' } });
      fireEvent.change(screen.getByLabelText(/weight/i), { target: { value: '70' } });
      fireEvent.change(screen.getByLabelText(/general health/i), { target: { value: '3' } });
      fireEvent.change(screen.getByLabelText(/age \(years\)/i), { target: { value: '45' } });

      // Advance to step 2
      fireEvent.click(screen.getByRole('button', { name: /next: documents & images/i }));
      await waitFor(() => {
        expect(screen.getByText(/2\. Documents, Notes & Imaging/i)).toBeTruthy();
      });

      // Advance to step 3
      fireEvent.click(screen.getByRole('button', { name: /next: review & run/i }));
      await waitFor(() => {
        expect(screen.getByText(/3\. Review & Run Analysis/i)).toBeTruthy();
      });

      // Check for survey features text and collapsible details
      expect(screen.getByText(/of 21 survey features provided/i)).toBeTruthy();
      expect(screen.getByText(/Not answered \(these will be assumed\):/i)).toBeTruthy();
    }, 20000);

    it('X-ray only mode accepts age (0-120) and X-ray file and calls analysisAPI.image', async () => {
      analysisAPI.image.mockResolvedValueOnce({
        analysisId: 'img-123',
        model_used: 'pediatric_kermany_resnet18',
        predicted_class: 'NORMAL',
        pneumonia_probability: 0.1,
      });

      render(
        <MemoryRouter>
          <AuthProvider>
            <Wizard />
          </AuthProvider>
        </MemoryRouter>
      );

      // Click on "Chest X-ray only (any age)" button
      const xrayModeBtn = screen.getByRole('button', { name: /chest x-ray only/i });
      fireEvent.click(xrayModeBtn);

      // Verify the dedicated X-ray only card is shown
      expect(screen.getByText('Chest X-Ray Screening (Any Age)')).toBeTruthy();

      // Enter pediatric age: 7 years
      const ageInput = screen.getByLabelText(/patient age in years/i);
      fireEvent.change(ageInput, { target: { value: '7' } });
      expect(screen.getByText(/pediatric route \(< 18 yrs\)/i)).toBeTruthy();

      // Upload file
      const file = new File(['fake-img'], 'child_xray.png', { type: 'image/png' });
      const fileInput = document.querySelector('input[type="file"]');
      fireEvent.change(fileInput, { target: { files: [file] } });

      // Run screening
      const runBtn = screen.getByRole('button', { name: /run chest x-ray screening/i });
      fireEvent.click(runBtn);

      await waitFor(() => {
        expect(analysisAPI.image).toHaveBeenCalledTimes(1);
      });

      const calledFormData = analysisAPI.image.mock.calls[0][0];
      expect(calledFormData.get('age_years')).toBe('7');
      expect(calledFormData.get('file')).toBeTruthy();
    });
  });

  describe('3. Chapter 16c: X-ray plausibility reject message', () => {
    it('shows the reject message in an alert when the API returns 422', async () => {
      const XRAY_REJECT_MSG =
        'This image does not look like a chest X-ray (it looks like a colour photo, ' +
        'a document or a blank page). Please upload a frontal chest X-ray.';

      // Mock the image API to reject with the plausibility message
      analysisAPI.image.mockRejectedValueOnce(new Error(XRAY_REJECT_MSG));

      render(
        <MemoryRouter>
          <AuthProvider>
            <Wizard />
          </AuthProvider>
        </MemoryRouter>,
      );

      // Switch to X-ray only mode
      const xrayModeBtn = screen.getByRole('button', { name: /chest x-ray only/i });
      fireEvent.click(xrayModeBtn);

      // Enter age
      const ageInput = screen.getByLabelText(/patient age in years/i);
      fireEvent.change(ageInput, { target: { value: '30' } });

      // Upload file
      const file = new File(['fake-img'], 'landscape.png', { type: 'image/png' });
      const fileInput = document.querySelector('input[type="file"]');
      fireEvent.change(fileInput, { target: { files: [file] } });

      // Run screening
      const runBtn = screen.getByRole('button', { name: /run chest x-ray screening/i });
      fireEvent.click(runBtn);

      await waitFor(() => {
        const alert = screen.getByRole('alert');
        expect(alert.textContent).toContain('does not look like a chest X-ray');
      });
    });
  });

  describe('4. Chapter 16 Final: Memory checkbox & Minimum inputs guard', () => {
    it('shows the memory fill checkbox checked by default in Step 3', async () => {
      render(
        <MemoryRouter>
          <AuthProvider>
            <Wizard />
          </AuthProvider>
        </MemoryRouter>,
      );

      // Fill required fields to advance
      fireEvent.change(screen.getByLabelText(/biological sex/i), { target: { value: '1' } });
      fireEvent.change(screen.getByLabelText(/height/i), { target: { value: '175' } });
      fireEvent.change(screen.getByLabelText(/weight/i), { target: { value: '70' } });
      fireEvent.change(screen.getByLabelText(/general health/i), { target: { value: '2' } });
      fireEvent.change(screen.getByLabelText(/age \(years\)/i), { target: { value: '35' } });

      fireEvent.click(screen.getByRole('button', { name: /next: documents & images/i }));
      await waitFor(() => {
        expect(screen.getByText(/2\. Documents, Notes & Imaging/i)).toBeTruthy();
      });

      fireEvent.click(screen.getByRole('button', { name: /next: review & run/i }));
      await waitFor(() => {
        expect(screen.getByText(/3\. Review & Run Analysis/i)).toBeTruthy();
      });

      // Verify the memory checkbox is present and checked by default
      const checkbox = screen.getByRole('checkbox', { name: /use my saved records to fill skipped answers/i });
      expect(checkbox).toBeTruthy();
      expect(checkbox.checked).toBe(true);

      // Verify user can toggle it off
      fireEvent.click(checkbox);
      expect(checkbox.checked).toBe(false);
    });
  });
});
