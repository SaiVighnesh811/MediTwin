import { describe, it, expect } from 'vitest';
import {
  mapAgeToBand,
  computeBMI,
  buildBRFSSPayload,
  formatFeature,
  formatFeatureName,
  formatFeatureInText,
  formatRuleTitle,
  formatCategory,
  parseExplanationItem,
  computeDivergingBars,
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
