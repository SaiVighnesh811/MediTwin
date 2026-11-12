import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AnalysisResultView from '../components/analysis/AnalysisResultView';

describe('Chapter 14, 14b & 14c: AnalysisResultView Component Tests', () => {
  it('1. low-reliability or Medium-class response shows the banner and never a confident single label', () => {
    const mockLowReliabilityResponse = {
      predicted_class: 'Medium',
      probabilities: { Low: 0.2, Medium: 0.55, High: 0.25 },
      completeness: '6 of 21 provided or extracted, 15 assumed',
      assumed_features: ['GenHlth', 'HighChol'],
      reliability: {
        level: 'low',
        reasons: [
          'Only 6 of 21 features were provided or extracted (fewer than 9).',
          "Predicted class is 'Medium' (Prediabetes), which has low precision and high uncertainty.",
        ],
      },
      limitations: ['Medium risk (prediabetes) class is weak/unreliable due to severe BRFSS class imbalance.'],
      recommendations: [
        {
          rule_id: 'REC_RISK_MEDIUM',
          category: 'risk_overall',
          text: 'Moderate risk screening indicator observed. Discuss lifestyle factors with your physician.',
        },
      ],
    };

    render(<AnalysisResultView data={mockLowReliabilityResponse} />);

    // 1. Reliability banner is rendered with "Input completeness" wording
    expect(screen.getByText(/Input completeness: low/i)).toBeTruthy();
    expect(screen.getByText(/describes how many inputs were available/i)).toBeTruthy();
    expect(screen.getByText(/fewer than 9/i)).toBeTruthy();

    // 2. Muted format is rendered, NOT confident label
    expect(screen.getByText(/Estimated Category:/i)).toBeTruthy();
    expect(screen.getByText(/High Uncertainty — see reliability banner/i)).toBeTruthy();
    expect(screen.queryByText(/^Class: Medium Risk$/i)).toBeNull();

    // 3. Probabilities rendered
    expect(screen.getByText(/55%/)).toBeTruthy();

    // 4. Uncertainty notice above recommendations is present
    expect(
      screen.getByText(/This estimate is uncertain because many answers were assumed. Treat this as general guidance, not a conclusion./i)
    ).toBeTruthy();
  });

  it('2. questionnaire-only input with skipped agents does NOT show amber alert notice, but shows neutral skipped details', () => {
    const mockSkippedOnlyResponse = {
      patient_id: 'P_FORM_ONLY',
      is_partial: true, // Orchestrator historically sets is_partial to true when skipped
      agent_status: {
        risk_agent: 'ok',
        recommendation_agent: 'ok',
        ocr_agent: 'skipped (no input)',
        nlp_agent: 'skipped (no input)',
        vision_agent: 'skipped (no input)',
      },
      risk_assessment: {
        predicted_class: 'High',
        probabilities: { Low: 0.1, Medium: 0.1, High: 0.8 },
      },
      reliability: { level: 'high', reasons: ['All questionnaire features present'] },
    };

    render(<AnalysisResultView data={mockSkippedOnlyResponse} />);

    // Amber partial alert must NOT be present
    expect(screen.queryByText(/Notice: Partial Pipeline Output/i)).toBeNull();

    // Neutral skipped list must be present
    expect(screen.getByText(/Not run: no input provided \(3 agents\)/i)).toBeTruthy();
    expect(screen.getByText(/ocr_agent/i)).toBeTruthy();
    expect(screen.getByText(/vision_agent/i)).toBeTruthy();
  });

  it('3. real agent failure displays amber alert notice naming the failed agent', () => {
    const mockFailedResponse = {
      patient_id: 'P_FAILED',
      is_partial: true,
      agent_status: {
        risk_agent: 'ok',
        nlp_agent: 'failed: DistilBERT token limit exceeded',
        ocr_agent: 'skipped (no input)',
      },
      risk_assessment: {
        predicted_class: 'High',
        probabilities: { Low: 0.1, Medium: 0.1, High: 0.8 },
      },
      reliability: { level: 'high', reasons: ['High feature count'] },
    };

    render(<AnalysisResultView data={mockFailedResponse} />);

    // Partial warning is rendered
    expect(screen.getByText(/Notice: Partial Pipeline Output/i)).toBeTruthy();
    expect(screen.getByText(/nlp_agent/i)).toBeTruthy();
    expect(screen.getByText(/DistilBERT token limit exceeded/i)).toBeTruthy();

    // Skipped agents also visible in neutral list
    expect(screen.getByText(/Not run: no input provided \(1 agent\)/i)).toBeTruthy();
  });

  it('4. translates raw BRFSS tokens to human-readable names in assumed features and SHAP factors', () => {
    const mockTranslationResponse = {
      risk_assessment: {
        predicted_class: 'Low',
        assumed_features: ['GenHlth', 'HighChol', 'DiffWalk'],
      },
      explainability: {
        top_shap_features: [
          'HighBP=1 increased the probability of High risk (SHAP=+0.1800)',
        ],
      },
    };

    render(<AnalysisResultView data={mockTranslationResponse} />);

    // Assumed features rendered in human-readable terms
    expect(screen.getByText(/General health, High cholesterol, Difficulty walking/i)).toBeTruthy();

    // SHAP factor rendered with human-readable label and text
    expect(screen.getByText('High blood pressure: Yes')).toBeTruthy();
    expect(screen.getByText(/High blood pressure: Yes increased the probability of High risk/i)).toBeTruthy();
  });

  it('5. renders human-readable recommendation titles, clean category tag, and NO action checkboxes', () => {
    const mockRecsResponse = {
      risk_assessment: { predicted_class: 'High' },
      recommendations: [
        {
          rule_id: 'REC_RISK_HIGH',
          category: 'risk_overall',
          text: 'Elevated risk profile detected. Prioritize scheduling an in-person clinical assessment with your physician.',
        },
        {
          rule_id: 'REC_BP_ELEVATED',
          category: 'BLOOD_PRESSURE',
          text: 'Blood pressure management: Adopt sodium reduction, DASH dietary patterns, and regular blood pressure monitoring.',
        },
      ],
    };

    render(<AnalysisResultView data={mockRecsResponse} />);

    // Rule titles mapped
    expect(screen.getByText('Elevated Metabolic Risk')).toBeTruthy();
    expect(screen.getByText('Blood Pressure Attention')).toBeTruthy();

    // Categories formatted
    expect(screen.getByText('Risk overall')).toBeTruthy();
    expect(screen.getByText('Blood pressure')).toBeTruthy();

    // Recommendations appear as the agent wrote them
    expect(screen.getByText(/Elevated risk profile detected/i)).toBeTruthy();
    expect(screen.getByText(/Blood pressure management: Adopt sodium reduction/i)).toBeTruthy();

    // CHAPTER 14c: No fake action checkboxes or split items
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
  });

  it('6. handles sparse response gracefully with fallback defaults', () => {
    const mockSparse = {
      risk_assessment: {},
    };

    const { container } = render(<AnalysisResultView data={mockSparse} />);
    expect(container).toBeTruthy();
    expect(screen.getByText(/CRITICAL DISCLAIMER/i)).toBeTruthy();
  });

  it('7. renders SHAP diverging bar chart with real string inputs, axis caption, and direction indicators', () => {
    const mockShapData = {
      risk_assessment: { predicted_class: 'High' },
      explainability: {
        top_shap_features: [
          'GenHlth=4 increased the probability of High risk (SHAP=+0.1368)',
          'HighBP=1 increased the probability of High risk (SHAP=+0.0516)',
          'BMI=27 lowered the probability of High risk (SHAP=-0.0285)',
        ],
      },
    };

    render(<AnalysisResultView data={mockShapData} />);

    // Axis caption
    expect(screen.getByText(/lowers High-risk probability/i)).toBeTruthy();
    expect(screen.getByText(/raises High-risk probability/i)).toBeTruthy();

    // Formatted readable feature labels at start of row
    expect(screen.getByText('General health: Fair')).toBeTruthy();
    expect(screen.getByText('High blood pressure: Yes')).toBeTruthy();
    expect(screen.getByText('BMI 27')).toBeTruthy();

    // Values with direction indicators at end of row
    expect(screen.getByText(/\+0\.1368 ↑/)).toBeTruthy();
    expect(screen.getByText(/\+0\.0516 ↑/)).toBeTruthy();
    expect(screen.getByText(/-0\.0285 ↓/)).toBeTruthy();

    // Explanatory footer line
    expect(
      screen.getByText(/Longer bars mean a stronger influence on this model's estimate. This describes the model, not medical causes./i)
    ).toBeTruthy();
  });

  it('8. renders LIME diverging bar chart when present in saved result (Case 3a)', () => {
    const mockLimeData = {
      risk_assessment: { predicted_class: 'High' },
      explainability: {
        top_shap_features: [
          'GenHlth=4 increased the probability of High risk (SHAP=+0.1368)',
        ],
        top_lime_features: [
          "Condition 'GenHlth > 3.00' pushes toward High risk (LIME weight=+0.1605)",
          "Condition 'BMI <= 25.00' decreases High risk (LIME weight=-0.0820)",
        ],
      },
    };

    render(<AnalysisResultView data={mockLimeData} />);

    // LIME section heading
    expect(screen.getByText(/Top Local Rule Influences \(LIME Local Surrogates\)/i)).toBeTruthy();

    // Condition labels
    expect(screen.getByText('General health > 3.00')).toBeTruthy();
    expect(screen.getByText('BMI <= 25.00')).toBeTruthy();

    // Weights with indicators
    expect(screen.getByText(/\+0\.1605 ↑/)).toBeTruthy();
    expect(screen.getByText(/-0\.0820 ↓/)).toBeTruthy();

    // Title reflects both SHAP & LIME
    expect(screen.getByText('Model Explainability (SHAP & LIME)')).toBeTruthy();
  });

  it('9. adult image card shows model score, related findings, limitations, caveat, and NO verdict words', () => {
    const mockAdultImage = {
      type: 'image',
      result: {
        model_name: 'densenet121-res224-all',
        model_used: 'adult_torchxrayvision_densenet121_all',
        population: 'Adult chest X-rays (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, RSNA)',
        pneumonia_score: 0.1425,
        pneumonia_op_thresh: 0.0568,
        related_findings: {
          Consolidation: 0.0812,
          'Lung Opacity': 0.1245,
          Infiltration: 0.0351,
        },
        limitations: [
          'Pretrained public model; no additional fine-tuning was performed in this project.',
          'Scores are op-normalized model outputs, not calibrated clinical probabilities.',
          'Screening aid only — not a diagnosis. A licensed radiologist must interpret all results.',
        ],
      },
    };

    render(<AnalysisResultView data={mockAdultImage} />);

    // 1. Title and model info
    expect(screen.getByText('Chest X-Ray Screening Aid')).toBeTruthy();
    expect(screen.getByText(/TorchXRayVision DenseNet121-res224-all/i)).toBeTruthy();

    // 2. Score gauge with "Pneumonia model score"
    expect(screen.getByText(/Pneumonia model score/i)).toBeTruthy();
    expect(screen.getByText(/14\.[23]%/)).toBeTruthy();

    // 3. Related findings
    expect(screen.getByText(/Related findings/i)).toBeTruthy();
    expect(screen.getByText(/Consolidation:/i)).toBeTruthy();
    expect(screen.getByText('8.1%')).toBeTruthy();
    expect(screen.getByText(/Lung Opacity:/i)).toBeTruthy();
    expect(screen.getByText(/12\.[45]%/)).toBeTruthy();

    // 4. Caveat line
    expect(
      screen.getByText(/Performance of this model was not independently measured in this project/i)
    ).toBeTruthy();

    // 5. Limitations list
    expect(screen.getByText(/Pretrained public model/i)).toBeTruthy();

    // 6. Must NOT contain verdict words
    expect(screen.queryByText(/pneumonia detected/i)).toBeNull();
    expect(screen.queryByText(/diagnosis positive/i)).toBeNull();

    // 7. No diabetes risk card for image-only analysis
    expect(screen.queryByText('Epidemiological Risk Stratification')).toBeNull();
  });

  it('10. pediatric image card preserves measured clinical metrics and finding', () => {
    const mockPediatricImage = {
      type: 'image',
      result: {
        model_used: 'pediatric_kermany_resnet18',
        predicted_class: 'NORMAL',
        pneumonia_probability: 0.142,
        normal_probability: 0.858,
        threshold_used: 0.5,
      },
    };

    render(<AnalysisResultView data={mockPediatricImage} />);

    // Finding
    expect(screen.getByText(/Screening Finding: NORMAL/i)).toBeTruthy();

    // Probability & threshold
    expect(screen.getByText(/Pneumonia probability: 14%/i)).toBeTruthy();
    expect(screen.getByText(/threshold: 0.5/i)).toBeTruthy();

    // Clinical caveat with 62.0% specificity
    expect(screen.getByText(/Official test specificity is 62.0%/i)).toBeTruthy();

    // No diabetes risk card for image-only analysis
    expect(screen.queryByText('Epidemiological Risk Stratification')).toBeNull();
  });

  it('11. standalone image analysis renders without risk, explanation or recommendation cards and without crashing', () => {
    const mockSparseImage = {
      type: 'image',
      result: {
        model_used: 'adult_torchxrayvision_densenet121_all',
        pneumonia_score: 0.05,
        disclaimer: 'Not for clinical use.',
      },
    };

    const { container } = render(<AnalysisResultView data={mockSparseImage} />);
    expect(container).toBeTruthy();

    // Only disclaimer and image card rendered
    expect(screen.getByText(/CRITICAL DISCLAIMER/i)).toBeTruthy();
    expect(screen.getByText('Chest X-Ray Screening Aid')).toBeTruthy();

    // Risk, explainability, recommendations, and doc cards not rendered
    expect(screen.queryByText('Epidemiological Risk Stratification')).toBeNull();
    expect(screen.queryByText(/Model Explainability/i)).toBeNull();
    expect(screen.queryByText(/Document Extraction/i)).toBeNull();
  });

  it('12. renders Document Extraction card with "No document uploaded" when empty, and extracted items when present', () => {
    // 1. Empty OCR
    const mockEmpty = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
      },
    };
    const { rerender } = render(<AnalysisResultView data={mockEmpty} />);
    expect(screen.getByText('Document Extraction (Prescription & Lab OCR)')).toBeTruthy();
    expect(screen.getByText(/No document uploaded/i)).toBeTruthy();

    // 2. Populated OCR
    const mockPopulated = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          extracted: [
            { type: 'prescription', medicine: 'Amlodipine', dosage: '500 mg', frequency: 'once daily' },
            { type: 'lab', test: 'HbA1c', value: '7.2', unit: '%' },
          ],
        },
      },
    };
    rerender(<AnalysisResultView data={mockPopulated} />);
    expect(screen.getByText('Amlodipine')).toBeTruthy();
    expect(screen.getByText(/500 mg/)).toBeTruthy();
    expect(screen.getByText('HbA1c')).toBeTruthy();
    expect(screen.getByText(/7\.2 %/)).toBeTruthy();
  });

  it('13. renders Biomedical Entities with merged words, grouped into categories, and Denied tag', () => {
    const mockNlp = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        nlp_entities: {
          diseases: [
            { text: 'hypertension', start: 0, end: 12, negated: false },
            { text: 'diabetes', start: 20, end: 28, negated: true },
          ],
          medications: [
            { text: 'cho', start: 50, end: 53, negated: false },
            { text: '##les', start: 53, end: 56, negated: false },
            { text: '##terol', start: 56, end: 61, negated: false },
          ],
        },
      },
    };

    render(<AnalysisResultView data={mockNlp} />);
    expect(screen.getByText('Biomedical Entities (Clinical Notes NER)')).toBeTruthy();
    expect(screen.getByText('Conditions (2)')).toBeTruthy();
    expect(screen.getByText('hypertension')).toBeTruthy();
    expect(screen.getByText('diabetes')).toBeTruthy();
    // Denied tag rendered on negated entity
    expect(screen.getByText('Denied')).toBeTruthy();
    // Merged subword medication rendered
    expect(screen.getByText('cholesterol')).toBeTruthy();
  });

  it('15. adult CXR score of 0.061 (6.1%) displays "Score below the model\'s operating point"', () => {
    const mockAdultBelow = {
      type: 'image',
      result: {
        model_name: 'densenet121-res224-all',
        model_used: 'adult_torchxrayvision_densenet121_all',
        population: 'Adult chest X-rays',
        pneumonia_score: 0.061,
        pneumonia_op_thresh: 0.0568,
      },
    };

    render(<AnalysisResultView data={mockAdultBelow} />);
    expect(screen.getByText(/Score below the model's operating point/i)).toBeTruthy();
    expect(screen.queryByText(/Score above the model's operating point/i)).toBeNull();
  });

  it('16. renders "dose not read" when medicine dosage is null in document OCR card', () => {
    const mockOcrNoDose = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [
            { medicine: 'Metformin', dosage: null, frequency: 'twice daily', duration: '30 days' },
          ],
        },
      },
    };

    render(<AnalysisResultView data={mockOcrNoDose} />);
    expect(screen.getByText('Read automatically from the image. Check against the original document.')).toBeTruthy();
    expect(screen.getByText('Metformin')).toBeTruthy();
    expect(screen.getByText(/dose not read/i)).toBeTruthy();
  });

  it('17. renders "No text could be read from the uploaded image." for empty OCR document and unclear flags', () => {
    // 1. Empty OCR document uploaded
    const mockEmptyDoc = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [],
          labs: [],
          raw_text: '',
          uploaded: true,
        },
      },
    };
    const { rerender } = render(<AnalysisResultView data={mockEmptyDoc} />);
    expect(screen.getByText('No text could be read from the uploaded image.')).toBeTruthy();
    expect(screen.queryByText('No document uploaded')).toBeNull();

    // 2. Unclear dose and unit flags
    const mockUnclear = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [
            { medicine: 'Amlodipine', dosage: null, dose_unclear: true, frequency: 'once daily', duration: '30 days' },
          ],
          labs: [
            { test: 'Total Cholesterol', value: '195', unit: null, unit_unclear: true, reference: '< 200', flag: 'within' },
          ],
        },
      },
    };
    rerender(<AnalysisResultView data={mockUnclear} />);
    expect(screen.getByText(/dose unclear - check the original/i)).toBeTruthy();
    expect(screen.getByText(/unit unclear/i)).toBeTruthy();
  });

  it('18. Chapter 16e: renders handwriting candidates above threshold and unclear with top3 below threshold', () => {
    const mockHw = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [],
          labs: [],
          raw_text: 'sample',
          handwriting_candidates: [
            {
              name: 'Aceta',
              probability: 0.85,
              top3: [
                { rank: 1, medicine_name: 'Aceta', probability: 0.85 },
                { rank: 2, medicine_name: 'Napa', probability: 0.08 },
                { rank: 3, medicine_name: 'Parpy', probability: 0.03 },
              ],
              source: 'handwriting_model',
            },
            {
              name: 'Unclear',
              probability: 0.22,
              top3: [
                { rank: 1, medicine_name: 'Amlodipine', probability: 0.22 },
                { rank: 2, medicine_name: 'Metformin', probability: 0.18 },
                { rank: 3, medicine_name: 'Losartan', probability: 0.15 },
              ],
              source: 'handwriting_model',
            },
          ],
        },
      },
    };

    render(<AnalysisResultView data={mockHw} />);
    expect(screen.getByText(/Possible medicine \(handwriting model, check the original\):/i)).toBeTruthy();
    expect(screen.getByText('Aceta')).toBeTruthy();
    expect(screen.getByText(/\(85%\)/)).toBeTruthy();

    expect(screen.getByText(/1 handwriting region was unclear/i)).toBeTruthy();
    expect(screen.getByText(/low confidence, do not rely on this: Amlodipine \(22%\), Metformin \(18%\), Losartan \(15%\)/i)).toBeTruthy();
  });

  it('19. Chapter 16f: renders collapsed unclear line, frequency unclear, and printed flag', () => {
    const mock16f = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [
            {
              medicine: 'Metformin',
              dosage: '500 mg',
              frequency: null,
              frequency_unclear: true,
              duration: '30 days',
            },
          ],
          labs: [
            {
              test: 'Fasting Blood Glucose',
              value: '110',
              unit: 'mg/dL',
              reference: '70 100',
              reference_inferred: true,
              flag: 'above',
              flag_printed: 'H',
            },
            {
              test: 'Total Cholesterol',
              value: '195',
              unit: 'mg/dL',
              reference: '200 ;',
              reference_unclear: true,
              flag: null,
            },
          ],
          raw_text: 'sample',
          unclear_regions: 3,
          handwriting_candidates: [],
        },
      },
    };

    render(<AnalysisResultView data={mock16f} />);
    expect(screen.getByText(/frequency unclear - check the original/i)).toBeTruthy();
    expect(screen.getByText(/printed flag: H/i)).toBeTruthy();
    expect(screen.getByText(/reference inferred: 70 100/i)).toBeTruthy();
    expect(screen.getByText(/reference unclear/i)).toBeTruthy();
    expect(screen.getByText(/3 handwriting regions were unclear/i)).toBeTruthy();
  });

  it('20. Chapter 16b: renders Family history chips and x-counts (x3)', () => {
    const mock16b = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        nlp_entities: {
          symptoms: [
            { text: 'cough', count: 3, negated: false },
          ],
          family_history: [
            { text: 'type 2 diabetes', count: 1, negated: false, subject: 'family' },
            { text: 'hypertension', count: 1, negated: false, subject: 'family' },
          ],
        },
      },
    };

    render(<AnalysisResultView data={mock16b} />);
    expect(screen.getByText('Biomedical Entities (Clinical Notes NER)')).toBeTruthy();
    expect(screen.getByText('Family History (2)')).toBeTruthy();
    expect(screen.getByText('type 2 diabetes')).toBeTruthy();
    expect(screen.getByText('hypertension')).toBeTruthy();
    expect(screen.getByText('Symptoms (1)')).toBeTruthy();
    expect(screen.getByText('cough')).toBeTruthy();
    expect(screen.getByText('x3')).toBeTruthy();
  });

  it('21. Chapter 16 Final: splits Input completeness and Model reliability banners, and renders saved records', () => {
    const mock16Final = {
      type: 'analyze',
      result: {
        risk_assessment: {
          predicted_class: 'Medium',
          probabilities: { Low: 0.1, Medium: 0.85, High: 0.05 },
          completeness: '19 of 21 provided or extracted, 2 assumed',
          saved_records: [
            { feature: 'HighBP', value: 1.0, saved_text: 'Amlodipine 5mg' },
            { feature: 'HighChol', value: 1.0, saved_text: 'Total Cholesterol: 240 mg/dL', date: '2025-05-01T10:00:00' },
          ],
        },
        reliability: {
          level: 'high',
          completeness_level: 'high',
          reasons: [
            '19 of 21 features were provided or extracted.',
            "Predicted class is 'Medium' (Prediabetes), which has low precision and high uncertainty.",
          ],
        },
      },
    };

    render(<AnalysisResultView data={mock16Final} />);

    // 1. Separate Input completeness banner showing high completeness
    expect(screen.getByText(/Input completeness: high/i)).toBeTruthy();

    // 2. Separate Model reliability banner showing Medium class low reliability
    expect(screen.getByText(/Medium class: low reliability/i)).toBeTruthy();

    // 3. Saved records listed under the exact specified title
    expect(screen.getByText('Filled from your saved records (not answered by you)')).toBeTruthy();
    expect(screen.getByText(/High blood pressure:/i)).toBeTruthy();
    expect(screen.getByText(/Amlodipine 5mg/i)).toBeTruthy();
    expect(screen.getByText(/Total Cholesterol: 240 mg\/dL/i)).toBeTruthy();
  });

  it('22. Chapter 16 Final: formatDuration normalizes "30 day" to "30 days" and keeps "1 day" singular', () => {
    const mockOcrDurations = {
      type: 'analyze',
      result: {
        risk_assessment: { predicted_class: 'Low', probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 } },
        reliability: { level: 'high', reasons: [] },
        ocr_findings: {
          medicines: [
            { medicine: 'Metformin', dosage: '500 mg', frequency: 'twice daily', duration: '30 day' },
            { medicine: 'Paracetamol', dosage: '500 mg', frequency: 'as needed', duration: '1 day' },
          ],
        },
      },
    };

    render(<AnalysisResultView data={mockOcrDurations} />);
    expect(screen.getByText(/30 days/i)).toBeTruthy();
    expect(screen.getByText(/1 day\b/i)).toBeTruthy();
  });

  it('23. Chapter 17: Medicine-aware guidance, "Based on:" lines, and inferred medicine markers', () => {
    const mock17 = {
      type: 'analyze',
      result: {
        risk_assessment: {
          predicted_class: 'Low',
          probabilities: { Low: 0.9, Medium: 0.05, High: 0.05 },
          saved_records: [
            {
              feature: 'HighBP',
              value: 1.0,
              saved_text: 'Medication: Amlodipine 5mg',
              date: '2026-11-10',
              inferred_from_medicine: true,
            },
          ],
        },
        reliability: { level: 'high', reasons: [] },
        recommendations: [
          {
            rule_id: 'REC_BP_HIGH',
            title: 'Blood Pressure Attention',
            text: 'Blood pressure management: Adopt sodium reduction, DASH dietary patterns.',
            based_on: 'saved record: high blood pressure = yes',
          },
          {
            rule_id: 'REC_MED_AMLODIPINE_0',
            title: 'Medicines found in your document',
            category: 'medicines',
            text: 'Amlodipine: dose unclear. Common use: blood pressure, and also chest pain. Extracted from your document - check it against the original and confirm with your doctor or pharmacist. Do not start, stop or change any medicine because of this page. Note: listed more than once; please check Note: You listed a medicine commonly used for blood pressure but answered No to blood pressure; please check your answers.',
            based_on: 'uploaded document: Amlodipine',
            notes: [
              'listed more than once; please check',
              'You listed a medicine commonly used for blood pressure but answered No to blood pressure; please check your answers.',
            ],
          },
        ],
      },
    };

    render(<AnalysisResultView data={mock17} />);

    // 1. "Based on:" lines rendered
    expect(screen.getAllByText(/Based on:/i).length).toBe(2);
    expect(screen.getByText(/saved record: high blood pressure = yes/i)).toBeTruthy();
    expect(screen.getByText(/uploaded document: Amlodipine/i)).toBeTruthy();

    // 2. Medicine card with "dose unclear"
    expect(screen.getByText(/Medicines found in your document/i)).toBeTruthy();
    expect(screen.getByText(/dose unclear/i)).toBeTruthy();
    expect(screen.getByText(/blood pressure, and also chest pain/i)).toBeTruthy();

    // 3. Consistency notes
    expect(screen.getByText(/listed more than once; please check/i)).toBeTruthy();
    expect(screen.getByText(/You listed a medicine commonly used for blood pressure but answered No to blood pressure; please check your answers./i)).toBeTruthy();

    // 4. Inferred from medicine label
    expect(screen.getByText(/\(inferred from a medicine; this medicine can also be used for other conditions, please verify\)/i)).toBeTruthy();
  });
});
