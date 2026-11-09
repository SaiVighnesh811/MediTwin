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

    // 1. Reliability banner is rendered
    expect(screen.getByText(/Reliability Assessment: low/i)).toBeTruthy();
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
});
