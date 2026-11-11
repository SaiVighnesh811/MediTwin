import { describe, it, expect } from 'vitest';
import {
  mapAgeToBand,
  computeBMI,
  buildBRFSSPayload,
  formatFeature,
  formatFeatureName,
  formatFeatureInText,
  formatCondition,
  formatRuleTitle,
  formatCategory,
  parseExplanationItem,
  computeDivergingBars,
  getGroupedEntities,
  parseOCRFindings,
  ALL_BRFSS_KEYS,
  FEATURE_NAMES,
} from '../lib/brfss';

describe('Chapter 14 & 14c: BRFSS Conversion, Presentation & Explanation Tests', () => {
  it('1. maps age boundary values correctly across all 13 bands', () => {
    // Under minimum
    expect(mapAgeToBand(17)).toBeNull();
    expect(mapAgeToBand(0)).toBeNull();
    expect(mapAgeToBand(-5)).toBeNull();

    // Band 1: 18 - 24
    expect(mapAgeToBand(18)).toBe(1);
    expect(mapAgeToBand(24)).toBe(1);

    // Band 2: 25 - 29
    expect(mapAgeToBand(25)).toBe(2);
    expect(mapAgeToBand(29)).toBe(2);

    // Band 3: 30 - 34
    expect(mapAgeToBand(30)).toBe(3);
    expect(mapAgeToBand(34)).toBe(3);

    // Band 4: 35 - 39
    expect(mapAgeToBand(35)).toBe(4);
    expect(mapAgeToBand(39)).toBe(4);

    // Band 5: 40 - 44
    expect(mapAgeToBand(40)).toBe(5);
    expect(mapAgeToBand(44)).toBe(5);

    // Band 6: 45 - 49
    expect(mapAgeToBand(45)).toBe(6);
    expect(mapAgeToBand(49)).toBe(6);

    // Band 7: 50 - 54
    expect(mapAgeToBand(50)).toBe(7);
    expect(mapAgeToBand(54)).toBe(7);

    // Band 8: 55 - 59
    expect(mapAgeToBand(55)).toBe(8);
    expect(mapAgeToBand(59)).toBe(8);

    // Band 9: 60 - 64
    expect(mapAgeToBand(60)).toBe(9);
    expect(mapAgeToBand(64)).toBe(9);

    // Band 10: 65 - 69
    expect(mapAgeToBand(65)).toBe(10);
    expect(mapAgeToBand(69)).toBe(10);

    // Band 11: 70 - 74
    expect(mapAgeToBand(70)).toBe(11);
    expect(mapAgeToBand(74)).toBe(11);

    // Band 12: 75 - 79
    expect(mapAgeToBand(75)).toBe(12);
    expect(mapAgeToBand(79)).toBe(12);

    // Band 13: 80+
    expect(mapAgeToBand(80)).toBe(13);
    expect(mapAgeToBand(95)).toBe(13);
    expect(mapAgeToBand(120)).toBe(13);

    // Over maximum
    expect(mapAgeToBand(121)).toBeNull();
  });

  it('2. computes BMI accurately from height and weight and rejects out-of-range inputs', () => {
    // Normal BMI: 175cm, 70kg -> 70 / (1.75)^2 = 22.857 -> 22.9
    expect(computeBMI(175, 70)).toBe(22.9);

    // High BMI: 160cm, 95kg -> 95 / (1.6)^2 = 37.109 -> 37.1
    expect(computeBMI(160, 95)).toBe(37.1);

    // Impossible height / weight
    expect(computeBMI(90, 70)).toBeNull(); // height < 100
    expect(computeBMI(260, 70)).toBeNull(); // height > 250
    expect(computeBMI(175, 20)).toBeNull(); // weight < 25
    expect(computeBMI(175, 350)).toBeNull(); // weight > 300

    // Missing
    expect(computeBMI('', 70)).toBeNull();
    expect(computeBMI(175, null)).toBeNull();
  });

  it('3. buildBRFSSPayload omits unselected/skipped fields and includes provided ones', () => {
    const rawForm = {
      age: 62,
      sex: 1,
      height: 180,
      weight: 80,
      genHlth: 2,
      HighBP: 1,
      HighChol: null, // skipped
      Smoker: 0,
      Stroke: undefined, // skipped
      mentHlth: 5,
      education: 6,
    };

    const payload = buildBRFSSPayload(rawForm);

    expect(payload.Age).toBe(9); // 62 is band 9
    expect(payload.Sex).toBe(1);
    expect(payload.BMI).toBe(24.7); // 80 / (1.8)^2 = 24.69
    expect(payload.GenHlth).toBe(2);
    expect(payload.HighBP).toBe(1.0);
    expect(payload.Smoker).toBe(0.0);
    expect(payload.MentHlth).toBe(5);
    expect(payload.Education).toBe(6);

    // Skipped fields MUST NOT be in payload
    expect(payload.HighChol).toBeUndefined();
    expect(payload.Stroke).toBeUndefined();
    expect(payload.Income).toBeUndefined();
    expect(payload.DiffWalk).toBeUndefined();
  });

  it('4. formatFeature maps raw feature values to human-readable strings', () => {
    // GenHlth 1-5
    expect(formatFeature('GenHlth', 4)).toBe('General health: Fair');
    expect(formatFeature('GenHlth', 1)).toBe('General health: Excellent');

    // Binary HighBP & DiffWalk
    expect(formatFeature('HighBP', 1)).toBe('High blood pressure: Yes');
    expect(formatFeature('HighBP', 0)).toBe('High blood pressure: No');
    expect(formatFeature('DiffWalk', 1)).toBe('Difficulty walking: Yes');
    expect(formatFeature('DiffWalk', 0)).toBe('Difficulty walking: No');

    // Age
    expect(formatFeature('Age', 9)).toBe('Age 60-64');
    expect(formatFeature('Age', 62)).toBe('Age 60-64');

    // BMI
    expect(formatFeature('BMI', 27)).toBe('BMI 27');
    expect(formatFeature('BMI', 28.4)).toBe('BMI 28.4');

    // Sex
    expect(formatFeature('Sex', 1)).toBe('Sex: Male');
    expect(formatFeature('Sex', 0)).toBe('Sex: Female');

    // Unvalued feature name lookup
    expect(formatFeature('GenHlth')).toBe('General health');
    expect(formatFeatureName('HighChol')).toBe('High cholesterol');
  });

  it('5. formatRuleTitle and formatCategory map rule codes and categories cleanly', () => {
    expect(formatRuleTitle('REC_RISK_HIGH')).toBe('Elevated Metabolic Risk');
    expect(formatRuleTitle('REC_RISK_MEDIUM')).toBe('Moderate Risk Screening Indicator');
    expect(formatRuleTitle('REC_BP_ELEVATED')).toBe('Blood Pressure Attention');
    expect(formatRuleTitle('REC_BP_HIGH')).toBe('Blood Pressure Attention');
    expect(formatRuleTitle('REC_LIFESTYLE_SMOKING')).toBe('Tobacco Cessation Support');
    expect(formatRuleTitle('REC_SMOKING')).toBe('Tobacco Cessation Support');
    expect(formatRuleTitle('REC_CHOLESTEROL')).toBe('Lipid Profile Attention');
    expect(formatRuleTitle('REC_WELLNESS_TRACKING')).toBe('Wellness Tracking');

    // Category formatting
    expect(formatCategory('BLOOD_PRESSURE')).toBe('Blood pressure');
    expect(formatCategory('LIFESTYLE')).toBe('Lifestyle');
    expect(formatCategory('risk_overall')).toBe('Risk overall');
  });

  it('6. formatFeatureInText converts <FeatureName>=<value> tokens to exact readable forms', () => {
    // Exact user examples
    expect(
      formatFeatureInText('GenHlth=4 increased the probability of High risk (SHAP=+0.1368)')
    ).toBe('General health: Fair increased the probability of High risk (SHAP=+0.1368)');

    expect(formatFeatureInText('HighBP=1')).toBe('High blood pressure: Yes');
    expect(formatFeatureInText('Age=9')).toBe('Age 60-64');
    expect(formatFeatureInText('BMI=27')).toBe('BMI 27');
    expect(formatFeatureInText('Sex=1')).toBe('Sex: Male');

    // Safe fallback for unrecognised token
    expect(formatFeatureInText('UnknownVar=999')).toBe('UnknownVar=999');
  });

  it('7. parseExplanationItem extracts numeric weights and labels from real string outputs', () => {
    const shapItem = 'GenHlth=4 increased the probability of High risk (SHAP=+0.1368)';
    const parsedShap = parseExplanationItem(shapItem, 'shap');
    expect(parsedShap.value).toBe(0.1368);
    expect(parsedShap.label).toBe('General health: Fair');
    expect(parsedShap.direction).toBe('right');
    expect(parsedShap.sentence).toContain('General health: Fair');

    const limeItem = "Condition 'GenHlth > 3.00' pushes toward High risk (LIME weight=+0.1605)";
    const parsedLime = parseExplanationItem(limeItem, 'lime');
    expect(parsedLime.value).toBe(0.1605);
    expect(parsedLime.label).toBe('General health > 3.00');
    expect(parsedLime.direction).toBe('right');
  });

  it('8. computeDivergingBars accurately orders directions and bar lengths from real example values', () => {
    const realExampleValues = [
      'GenHlth=4 increased the probability of High risk (SHAP=+0.1368)',
      'HighBP=1 increased the probability of High risk (SHAP=+0.0516)',
      'Age=9 increased the probability of High risk (SHAP=+0.0440)',
      'BMI=27 lowered the probability of High risk (SHAP=-0.0285)',
      'PhysActivity=0 increased the probability of High risk (SHAP=+0.0171)',
    ];

    const bars = computeDivergingBars(realExampleValues, 'shap');
    expect(bars).toHaveLength(5);

    // Max absolute is 0.1368 -> largest bar width is 50%
    expect(bars[0].widthPct).toBe(50);
    expect(bars[0].direction).toBe('right');
    expect(bars[0].formattedValue).toBe('+0.1368');
    expect(bars[0].indicator).toBe('↑');

    // Row 1: +0.0516 -> (0.0516 / 0.1368) * 50 = ~18.9%
    expect(bars[1].widthPct).toBe(18.9);
    expect(bars[1].direction).toBe('right');
    expect(bars[1].indicator).toBe('↑');

    // Row 2: +0.0440 -> (0.0440 / 0.1368) * 50 = ~16.1%
    expect(bars[2].widthPct).toBe(16.1);
    expect(bars[2].direction).toBe('right');
    expect(bars[2].indicator).toBe('↑');

    // Row 3: -0.0285 -> left direction, (0.0285 / 0.1368) * 50 = ~10.4%
    expect(bars[3].widthPct).toBe(10.4);
    expect(bars[3].direction).toBe('left');
    expect(bars[3].formattedValue).toBe('-0.0285');
    expect(bars[3].indicator).toBe('↓');

    // Row 4: +0.0171 -> (0.0171 / 0.1368) * 50 = ~6.3%
    expect(bars[4].widthPct).toBe(6.3);
    expect(bars[4].direction).toBe('right');
    expect(bars[4].indicator).toBe('↑');

    // All rows strictly respect minimum visible width of 4%
    bars.forEach((b) => {
      expect(b.widthPct).toBeGreaterThanOrEqual(4);
    });

    // Length ordering strictly matches |value|
    for (let i = 0; i < bars.length - 1; i++) {
      expect(bars[i].widthPct).toBeGreaterThanOrEqual(bars[i + 1].widthPct);
    }
  });
});

// ─────────────────────────────────────────────────────────────
// Chapter 15a: ALL_BRFSS_KEYS, full payload, MentHlth/PhysHlth edge cases
// ─────────────────────────────────────────────────────────────
describe('Chapter 15a: buildBRFSSPayload all 21 features and edge cases', () => {
  it('9. ALL_BRFSS_KEYS exports exactly 21 keys matching FEATURE_NAMES', () => {
    expect(ALL_BRFSS_KEYS).toHaveLength(21);
    // Every key must exist in FEATURE_NAMES
    ALL_BRFSS_KEYS.forEach((k) => {
      expect(FEATURE_NAMES[k]).toBeDefined();
    });
    // No duplicates
    expect(new Set(ALL_BRFSS_KEYS).size).toBe(21);
  });

  it('10. buildBRFSSPayload with all 21 fields answered produces exactly 21 keys', () => {
    const fullForm = {
      age: 35,       // -> Age: 4
      sex: 1,        // -> Sex: 1
      height: 175,
      weight: 70,    // -> BMI: 22.9
      genHlth: 3,    // -> GenHlth: 3
      HighBP: 1,
      HighChol: 0,
      CholCheck: 1,
      Smoker: 0,
      Stroke: 0,
      HeartDiseaseorAttack: 0,
      PhysActivity: 1,
      Fruits: 1,
      Veggies: 1,
      HvyAlcoholConsump: 0,
      AnyHealthcare: 1,
      NoDocbcCost: 0,
      DiffWalk: 0,
      mentHlth: 0,   // -> MentHlth: 0  (0 must be valid!)
      physHlth: 0,   // -> PhysHlth: 0  (0 must be valid!)
      education: 5,  // -> Education: 5
      income: 6,     // -> Income: 6
    };

    const payload = buildBRFSSPayload(fullForm);
    const keys = Object.keys(payload);

    // Must have exactly 21 keys
    expect(keys).toHaveLength(21);

    // Spot-check individual values
    expect(payload.Age).toBe(4);       // 35 -> band 4
    expect(payload.Sex).toBe(1);
    expect(payload.BMI).toBe(22.9);
    expect(payload.GenHlth).toBe(3);
    expect(payload.HighBP).toBe(1.0);
    expect(payload.HighChol).toBe(0.0);
    expect(payload.MentHlth).toBe(0);  // 0 is valid!
    expect(payload.PhysHlth).toBe(0);  // 0 is valid!
    expect(payload.Education).toBe(5);
    expect(payload.Income).toBe(6);
  });

  it('11. mentHlth/physHlth accept string "0" as 0 (valid, not omitted)', () => {
    const form = {
      age: 25, sex: 0, height: 165, weight: 65, genHlth: 2,
      mentHlth: '0',
      physHlth: '0',
    };
    const payload = buildBRFSSPayload(form);
    expect(payload.MentHlth).toBe(0);
    expect(payload.PhysHlth).toBe(0);
  });

  it('12. mentHlth/physHlth accept both camelCase and PascalCase keys', () => {
    // PascalCase form (e.g. already-processed data)
    const formPascal = {
      Age: 5, Sex: 1, BMI: 25.0, GenHlth: 2,
      MentHlth: 10,
      PhysHlth: 5,
    };
    const payload = buildBRFSSPayload(formPascal);
    expect(payload.MentHlth).toBe(10);
    expect(payload.PhysHlth).toBe(5);
  });

  it('13. unanswered keys are exactly ALL_BRFSS_KEYS minus answered payload keys', () => {
    // Only 4 required fields filled
    const minimalForm = {
      age: 50, sex: 1, height: 170, weight: 75, genHlth: 4,
    };
    const payload = buildBRFSSPayload(minimalForm);
    const answeredCount = Object.keys(payload).length;
    expect(answeredCount).toBeLessThan(21);

    const unansweredKeys = ALL_BRFSS_KEYS.filter((k) => !(k in payload));
    // All unanswered keys must have FEATURE_NAMES labels
    unansweredKeys.forEach((k) => {
      expect(FEATURE_NAMES[k]).toBeDefined();
    });
    // Total must add up to 21
    expect(answeredCount + unansweredKeys.length).toBe(21);
  });
});


// ─────────────────────────────────────────────────────────────
// Chapter 15b: LIME Condition Formatting, NLP Merging & OCR Extraction
// ─────────────────────────────────────────────────────────────
describe('Chapter 15b: LIME Condition Formatting, NLP Merging & OCR Extraction', () => {
  it('14. formats binary rules (> -> Yes, <= -> No) and non-binary rules cleanly', () => {
    // Binary: HighBP > 0.00 -> Yes, <= 0.00 -> No
    expect(formatCondition('HighBP > 0.00')).toBe('High blood pressure: Yes');
    expect(formatCondition('HighBP <= 0.00')).toBe('High blood pressure: No');
    expect(formatFeatureInText('HighBP > 0.00')).toBe('High blood pressure: Yes');
    expect(formatFeatureInText('HighBP <= 0.00')).toBe('High blood pressure: No');
    expect(formatRuleTitle('HighBP > 0.00')).toBe('High blood pressure: Yes');
    expect(formatRuleTitle('HighBP <= 0.00')).toBe('High blood pressure: No');

    // Other binary features: Smoker, DiffWalk, HighChol
    expect(formatCondition('Smoker > 0.00')).toBe('Smoking history: Yes');
    expect(formatCondition('Smoker <= 0.00')).toBe('Smoking history: No');
    expect(formatCondition('DiffWalk > 0.00')).toBe('Difficulty walking: Yes');
    expect(formatCondition('DiffWalk <= 0.00')).toBe('Difficulty walking: No');

    // Non-binary: GenHlth > 3.00, BMI <= 25.00
    expect(formatCondition('GenHlth > 3.00')).toBe('General health > 3.00');
    expect(formatFeatureInText('GenHlth > 3.00')).toBe('General health > 3.00');
    expect(formatRuleTitle('GenHlth > 3.00')).toBe('General health > 3.00');
    expect(formatCondition('BMI <= 25.00')).toBe('BMI <= 25.00');
  });

  it('15. LIME sentences close quotes and heading matches sentence text', () => {
    // Binary condition in LIME sentence
    const limeItemYes = "Condition 'HighBP > 0.00' pushes toward High risk (LIME weight=+0.1605)";
    const parsedYes = parseExplanationItem(limeItemYes, 'lime');
    expect(parsedYes.label).toBe('High blood pressure: Yes');
    expect(parsedYes.sentence).toBe("Condition 'High blood pressure: Yes' pushes toward High risk (LIME weight=+0.1605)");

    const limeItemNo = "Condition 'HighBP <= 0.00' pushes away from High risk (LIME weight=-0.0500)";
    const parsedNo = parseExplanationItem(limeItemNo, 'lime');
    expect(parsedNo.label).toBe('High blood pressure: No');
    expect(parsedNo.sentence).toBe("Condition 'High blood pressure: No' pushes away from High risk (LIME weight=-0.0500)");

    // Non-binary condition in LIME sentence
    const limeItemGen = "Condition 'GenHlth > 3.00' pushes toward High risk (LIME weight=+0.1605)";
    const parsedGen = parseExplanationItem(limeItemGen, 'lime');
    expect(parsedGen.label).toBe('General health > 3.00');
    expect(parsedGen.sentence).toBe("Condition 'General health > 3.00' pushes toward High risk (LIME weight=+0.1605)");

    // Handles previously unclosed quote gracefully and closes it
    const unclosed = "Condition 'High blood pressure: No pushes toward High risk";
    const formattedUnclosed = formatFeatureInText(unclosed);
    expect(formattedUnclosed).toBe("Condition 'High blood pressure: No' pushes toward High risk");
  });

  it('16. getGroupedEntities joins word pieces, hides OTHER, groups into categories, and tags negated entities', () => {
    const rawNlp = {
      diseases: [
        { text: 'hypertension', start: 0, end: 12, negated: false },
        { text: 'diabetes', start: 20, end: 28, negated: true },
      ],
      symptoms: [
        { text: 'chest', start: 30, end: 35, negated: false },
        { text: 'pain', start: 36, end: 40, negated: false },
      ],
      medications: [
        { text: 'cho', start: 50, end: 53, negated: false },
        { text: '##les', start: 53, end: 56, negated: false },
        { text: '##terol', start: 56, end: 61, negated: false },
      ],
      procedures: [
        { text: 'X-ray', start: 70, end: 75, negated: false },
      ],
      entities: [
        // Subwords
        { text: 'cho', group: 'MEDICATION', entity_type: 'Medication', start: 50, end: 53, negated: false },
        { text: '##les', group: 'MEDICATION', entity_type: 'Medication', start: 53, end: 56, negated: false },
        { text: '##terol', group: 'MEDICATION', entity_type: 'Medication', start: 56, end: 61, negated: false },
        // An OTHER entity that must be hidden
        { text: '45 years old', group: 'OTHER', entity_type: 'Age', start: 80, end: 92, negated: false },
      ],
    };

    const grouped = getGroupedEntities(rawNlp);

    // Grouping checks
    expect(grouped.Condition).toHaveLength(2);
    expect(grouped.Condition[0].text).toBe('hypertension');
    expect(grouped.Condition[0].negated).toBe(false);
    expect(grouped.Condition[1].text).toBe('diabetes');
    expect(grouped.Condition[1].negated).toBe(true);

    // Subword merging: "cho" + "##les" + "##terol" -> "cholesterol"
    expect(grouped.Medication).toHaveLength(1);
    expect(grouped.Medication[0].text).toBe('cholesterol');

    // Procedure
    expect(grouped.Procedure).toHaveLength(1);
    expect(grouped.Procedure[0].text).toBe('X-ray');

    // OTHER entity must NOT be present in any group
    const allTexts = [
      ...grouped.Condition,
      ...grouped.Medication,
      ...grouped.Symptom,
      ...grouped.Procedure,
    ].map((e) => e.text);
    expect(allTexts).not.toContain('45 years old');
  });

  it('17. parseOCRFindings extracts medicines, lab values, and handles empty input', () => {
    // Empty / null OCR
    const emptyOcr = parseOCRFindings(null);
    expect(emptyOcr.hasOcr).toBe(false);
    expect(emptyOcr.medicines).toEqual([]);
    expect(emptyOcr.labValues).toEqual([]);

    // Real OCR shape with prescription and lab
    const populatedOcr = {
      printed_text: 'Rx: Metformin 500mg daily\nLab: HbA1c 7.2%',
      extracted: [
        { type: 'prescription', medicine: 'Metformin', dosage: '500 mg', frequency: 'daily' },
        { type: 'lab', test: 'HbA1c', value: '7.2', unit: '%', reference: '< 5.7' },
      ],
      handwriting_top3: [
        { medicine: 'Amlodipine', confidence: 0.92 },
      ],
    };

    const parsed = parseOCRFindings(populatedOcr);
    expect(parsed.hasOcr).toBe(true);
    expect(parsed.rawText).toContain('Metformin');
    expect(parsed.medicines).toHaveLength(1);
    expect(parsed.medicines[0].medicine).toBe('Metformin');
    expect(parsed.labValues).toHaveLength(1);
    expect(parsed.labValues[0].test).toBe('HbA1c');
    expect(parsed.handwriting).toHaveLength(1);
    expect(parsed.handwriting[0].medicine).toBe('Amlodipine');
  });
});
