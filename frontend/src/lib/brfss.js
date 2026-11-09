/**
 * src/lib/brfss.js
 * 
 * Pure BRFSS feature conversion, presentation, and explanation parser helpers.
 * Strictly adheres to BRFSS 2015 ranges, ml_service/schemas.py, and recommendation rules.
 */

// 1. Age band mapping (18-120 -> 1-13)
// 1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49,
// 7: 50-54, 8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80+
export const AGE_BANDS = {
  1: '18-24',
  2: '25-29',
  3: '30-34',
  4: '35-39',
  5: '40-44',
  6: '45-49',
  7: '50-54',
  8: '55-59',
  9: '60-64',
  10: '65-69',
  11: '70-74',
  12: '75-79',
  13: '80+',
};

export function mapAgeToBand(age) {
  if (typeof age !== 'number' || isNaN(age)) return null;
  age = Math.floor(age);
  if (age < 18 || age > 120) return null;
  if (age <= 24) return 1;
  if (age <= 29) return 2;
  if (age <= 34) return 3;
  if (age <= 39) return 4;
  if (age <= 44) return 5;
  if (age <= 49) return 6;
  if (age <= 54) return 7;
  if (age <= 59) return 8;
  if (age <= 64) return 9;
  if (age <= 69) return 10;
  if (age <= 74) return 11;
  if (age <= 79) return 12;
  return 13;
}

// 2. BMI calculation from height (cm) and weight (kg)
// BMI = weight / (height / 100)^2, rounded to 1 decimal place
// Validated range in ML schema: 12.0 to 98.0
export function computeBMI(heightCm, weightKg) {
  if (!heightCm || !weightKg) return null;
  const h = Number(heightCm);
  const w = Number(weightKg);
  if (isNaN(h) || isNaN(w)) return null;
  if (h < 100 || h > 250 || w < 25 || w > 300) return null;

  const heightM = h / 100;
  const bmi = w / (heightM * heightM);
  const rounded = Math.round(bmi * 10) / 10;
  if (rounded < 12.0 || rounded > 98.0) return null;
  return rounded;
}

// 3. General health mapping: 1 (Excellent) to 5 (Poor)
export const GEN_HLTH_OPTIONS = [
  { value: 1, label: 'Excellent' },
  { value: 2, label: 'Very good' },
  { value: 3, label: 'Good' },
  { value: 4, label: 'Fair' },
  { value: 5, label: 'Poor' },
];

// 4. Sex mapping: 0 female, 1 male
export const SEX_OPTIONS = [
  { value: 0, label: 'Female' },
  { value: 1, label: 'Male' },
];

// 5. Education level mapping: 1-6
export const EDUCATION_OPTIONS = [
  { value: 1, label: 'Never attended / kindergarten only' },
  { value: 2, label: 'Grades 1–8 (Elementary)' },
  { value: 3, label: 'Grades 9–11 (Some high school)' },
  { value: 4, label: 'Grade 12 or GED (High school grad)' },
  { value: 5, label: 'College 1–3 years (Some college / technical)' },
  { value: 6, label: 'College 4+ years (College graduate)' },
];

// 6. Income band mapping: 1-8
export const INCOME_OPTIONS = [
  { value: 1, label: 'Under $10,000' },
  { value: 2, label: '$10,000 – $14,999' },
  { value: 3, label: '$15,000 – $19,999' },
  { value: 4, label: '$20,000 – $24,999' },
  { value: 5, label: '$25,000 – $34,999' },
  { value: 6, label: '$35,000 – $49,999' },
  { value: 7, label: '$50,000 – $74,999' },
  { value: 8, label: '$75,000 or more' },
];

// 7. Binary yes/no fields list (3-state in UI: true -> 1.0, false -> 0.0, null/skip -> omit)
export const BINARY_FIELDS = [
  { key: 'HighBP', label: 'High blood pressure diagnosis' },
  { key: 'HighChol', label: 'High blood cholesterol diagnosis' },
  { key: 'CholCheck', label: 'Cholesterol check in past 5 years' },
  { key: 'Smoker', label: 'Smoked at least 100 cigarettes in life' },
  { key: 'Stroke', label: 'Ever told you had a stroke' },
  { key: 'HeartDiseaseorAttack', label: 'Coronary heart disease or heart attack' },
  { key: 'PhysActivity', label: 'Physical activity / exercise in past 30 days' },
  { key: 'Fruits', label: 'Consume fruit 1+ times per day' },
  { key: 'Veggies', label: 'Consume vegetables 1+ times per day' },
  { key: 'HvyAlcoholConsump', label: 'Heavy alcohol consumption (men >14/wk, women >7/wk)' },
  { key: 'AnyHealthcare', label: 'Any health care coverage / insurance' },
  { key: 'NoDocbcCost', label: 'Needed to see doctor in past 12 mo but could not due to cost' },
  { key: 'DiffWalk', label: 'Serious difficulty walking or climbing stairs' },
];

// 8. Human-friendly feature name dictionary for all 21 BRFSS features
export const FEATURE_NAMES = {
  GenHlth: 'General health',
  HighBP: 'High blood pressure',
  HighChol: 'High cholesterol',
  CholCheck: 'Cholesterol check in past 5 years',
  BMI: 'BMI',
  Smoker: 'Smoking history',
  Stroke: 'History of stroke',
  HeartDiseaseorAttack: 'Coronary heart disease / heart attack',
  PhysActivity: 'Physical activity in past 30 days',
  Fruits: 'Daily fruit consumption',
  Veggies: 'Daily vegetable consumption',
  HvyAlcoholConsump: 'Heavy alcohol consumption',
  AnyHealthcare: 'Health insurance coverage',
  NoDocbcCost: 'Doctor visit cost barrier',
  MentHlth: 'Days of poor mental health',
  PhysHlth: 'Days of poor physical health',
  DiffWalk: 'Difficulty walking',
  Sex: 'Sex',
  Age: 'Age',
  Education: 'Education level',
  Income: 'Income level',
};

// Case-insensitive lookup map mapping keys and friendly names back to standard key
export const NAME_TO_KEY = {};
for (const [key, name] of Object.entries(FEATURE_NAMES)) {
  NAME_TO_KEY[key.toLowerCase()] = key;
  NAME_TO_KEY[name.toLowerCase()] = key;
}

export function formatFeatureName(name) {
  if (!name) return '';
  return FEATURE_NAMES[name] || name;
}

/**
 * Format a feature name and optional value into a user-friendly string.
 * Example outputs:
 * - formatFeature('GenHlth', 4) -> "General health: Fair"
 * - formatFeature('HighBP', 1) -> "High blood pressure: Yes"
 * - formatFeature('HighBP', 0) -> "High blood pressure: No"
 * - formatFeature('Age', 9) -> "Age 60-64"
 * - formatFeature('Age', 62) -> "Age 60-64"
 * - formatFeature('DiffWalk', 1) -> "Difficulty walking: Yes"
 * - formatFeature('BMI', 27) -> "BMI 27"
 * - formatFeature('Sex', 1) -> "Sex: Male"
 * - formatFeature('GenHlth') -> "General health"
 */
export function formatFeature(name, value) {
  if (!name) return '';
  if (value === undefined || value === null) {
    return formatFeatureName(name);
  }

  // 1. General health
  if (name === 'GenHlth') {
    const opt = GEN_HLTH_OPTIONS.find((o) => o.value === Number(value));
    return `General health: ${opt ? opt.label : value}`;
  }

  // 2. High Blood Pressure
  if (name === 'HighBP') {
    const isYes = Number(value) === 1 || value === true || value === '1';
    return `High blood pressure: ${isYes ? 'Yes' : 'No'}`;
  }

  // 3. Difficulty Walking
  if (name === 'DiffWalk') {
    const isYes = Number(value) === 1 || value === true || value === '1';
    return `Difficulty walking: ${isYes ? 'Yes' : 'No'}`;
  }

  // 4. BMI
  if (name === 'BMI') {
    const n = Number(value);
    const formatted = !isNaN(n) ? (Number.isInteger(n) ? n.toString() : n.toFixed(1)) : value;
    return `BMI ${formatted}`;
  }

  // 5. Age (band 1-13 or raw age number)
  if (name === 'Age') {
    const n = Number(value);
    let bandNum = n;
    if (n >= 18) {
      bandNum = mapAgeToBand(n);
    }
    const bandLabel = AGE_BANDS[bandNum];
    if (bandLabel) {
      return `Age ${bandLabel}`;
    }
    return `Age ${value}`;
  }

  // 6. Sex
  if (name === 'Sex') {
    return `Sex: ${Number(value) === 1 ? 'Male' : 'Female'}`;
  }

  // 7. Days (MentHlth, PhysHlth)
  if (name === 'MentHlth') {
    return `Days of poor mental health: ${value}`;
  }
  if (name === 'PhysHlth') {
    return `Days of poor physical health: ${value}`;
  }

  // 8. Education
  if (name === 'Education') {
    const opt = EDUCATION_OPTIONS.find((o) => o.value === Number(value));
    return `Education: ${opt ? opt.label : value}`;
  }

  // 9. Income
  if (name === 'Income') {
    const opt = INCOME_OPTIONS.find((o) => o.value === Number(value));
    return `Income: ${opt ? opt.label : value}`;
  }

  // 10. Binary fields
  const isBinary = BINARY_FIELDS.some((b) => b.key === name);
  if (isBinary) {
    const isYes = Number(value) === 1 || value === true || value === '1';
    return `${formatFeatureName(name)}: ${isYes ? 'Yes' : 'No'}`;
  }

  return `${formatFeatureName(name)}: ${value}`;
}

/**
 * Replace raw feature tokens and <FeatureName>=<value> patterns in an explanation string
 * with human-readable labels and formatted values.
 * e.g. "GenHlth=4 increased the probability of High risk (SHAP=+0.1368)"
 *   -> "General health: Fair increased the probability of High risk (SHAP=+0.1368)"
 */
export function formatFeatureInText(text) {
  if (typeof text !== 'string') return text;
  let out = text;

  // 1. Convert <FeatureName>=<value> tokens first
  const pattern = /([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)*)=([^\s,;()]+)/g;
  out = out.replace(pattern, (match, featName, val) => {
    try {
      const cleanName = featName.trim();
      const standardKey = NAME_TO_KEY[cleanName.toLowerCase()] || (FEATURE_NAMES[cleanName] ? cleanName : null);
      if (standardKey) {
        return formatFeature(standardKey, val);
      }
      return match;
    } catch {
      return match;
    }
  });

  // 2. Replace standalone feature keys with friendly names
  for (const [key, name] of Object.entries(FEATURE_NAMES)) {
    const reg = new RegExp(`\\b${key}\\b`, 'g');
    out = out.replace(reg, name);
  }

  return out;
}

// 9. Human-readable titles for recommendation rule codes
export const RULE_TITLES = {
  REC_RISK_HIGH: 'Elevated Metabolic Risk',
  REC_RISK_MEDIUM: 'Moderate Risk Screening Indicator',
  REC_BP_HIGH: 'Blood Pressure Attention',
  REC_BP_ELEVATED: 'Blood Pressure Attention',
  REC_CHOLESTEROL: 'Lipid Profile Attention',
  REC_CHOL_CHECK: 'Cholesterol Screening Due',
  REC_PHYSICAL_ACTIVITY: 'Physical Activity Guideline',
  REC_LIFESTYLE_PHYSICAL: 'Physical Activity Guideline',
  REC_BMI_NUTRITION: 'Dietary Nutrition Guideline',
  REC_LIFESTYLE_NUTRITION: 'Dietary Nutrition Guideline',
  REC_SMOKING: 'Tobacco Cessation Support',
  REC_LIFESTYLE_SMOKING: 'Tobacco Cessation Support',
  REC_GLYCEMIC_LAB: 'Glycemic & Metabolic Monitoring',
  REC_IMAGING_ABNORMAL: 'Chest Imaging Clinical Follow-up',
  REC_PNEUMONIA_FINDING: 'Chest Imaging Clinical Follow-up',
  REC_ROUTINE_CHECKUP: 'Routine Wellness & Preventive Checkup',
  REC_DOC_COST: 'Healthcare Access & Resources',
};

export function formatRuleTitle(ruleId) {
  if (!ruleId) return 'Guidance Recommendation';
  if (RULE_TITLES[ruleId]) return RULE_TITLES[ruleId];
  return ruleId
    .replace(/^REC_/, '')
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

/**
 * Format category tag into clean title case e.g. BLOOD_PRESSURE -> Blood pressure
 */
export function formatCategory(cat) {
  if (!cat) return '';
  const clean = String(cat).replace(/[_-]+/g, ' ').trim().toLowerCase();
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

/**
 * Parses an explanation item (SHAP or LIME), whether a string or an object,
 * extracting the numeric influence value, formatted label with value, and clean sentence.
 */
export function parseExplanationItem(item, type = 'shap') {
  if (!item) return null;

  // Object shape: { feature, shap/weight, value, sentence }
  if (typeof item === 'object') {
    const val = typeof item.shap === 'number'
      ? item.shap
      : (typeof item.weight === 'number'
        ? item.weight
        : (parseFloat(item.shap || item.weight || item.value || 0) || 0));

    let label = item.label || '';
    if (!label && item.feature) {
      if (item.value !== undefined && item.value !== null) {
        label = formatFeature(item.feature, item.value);
      } else {
        label = formatFeatureInText(formatFeatureName(item.feature));
      }
    }
    const sentence = formatFeatureInText(item.sentence || item.text || label);

    return {
      feature: item.feature || '',
      label: label || 'Factor',
      value: val,
      sentence,
      direction: val >= 0 ? 'right' : 'left',
    };
  }

  // String shape: e.g. "GenHlth=4 increased the probability of High risk (SHAP=+0.1368)"
  // or "Condition 'GenHlth > 3.00' pushes toward High risk (LIME weight=+0.1605)"
  const str = String(item).trim();

  // 1. Extract numeric value
  let val = 0;
  const numMatch = str.match(/(?:SHAP|weight|LIME weight)=([+-]?\d+(?:\.\d+)?)/i);
  if (numMatch) {
    val = parseFloat(numMatch[1]);
  }

  // 2. Extract feature / label
  let label = '';
  let feature = '';

  // Check for condition in quotes first (standard LIME output)
  const condMatch = str.match(/'([^']+)'/);
  if (condMatch) {
    feature = condMatch[1];
    label = formatFeatureInText(condMatch[1]);
  } else {
    // Check for "<FeatureName>=<value>" at the beginning of sentence (standard SHAP output)
    const eqMatch = str.match(/^([A-Za-z0-9_]+)=([^\s,;()]+)/);
    if (eqMatch) {
      feature = eqMatch[1];
      label = formatFeature(feature, eqMatch[2]);
    } else {
      // Find any known BRFSS feature with =
      const anyEq = str.match(/([A-Za-z0-9_]+)=([^\s,;()]+)/g);
      let foundKnown = false;
      if (anyEq) {
        for (const token of anyEq) {
          const [k, v] = token.split('=');
          if (NAME_TO_KEY[k.toLowerCase()]) {
            feature = k;
            label = formatFeature(k, v);
            foundKnown = true;
            break;
          }
        }
      }
      if (!foundKnown) {
        // Fallback: take leading words before verb
        label = formatFeatureInText(str.split(/\s+(?:increased|lowered|pushes|decreases)/i)[0]);
      }
    }
  }

  const sentence = formatFeatureInText(str);

  return {
    feature,
    label: label || 'Factor',
    value: val,
    sentence,
    direction: val >= 0 ? 'right' : 'left',
  };
}

/**
 * Computes diverging bar metrics for a list of SHAP or LIME explanation items.
 * Each bar starts at center (50%), goes right if value >= 0, left if value < 0.
 * Width is proportional to |value| / maxAbsValue (scaled to max 50%, min 4%).
 */
export function computeDivergingBars(items, type = 'shap') {
  if (!Array.isArray(items) || items.length === 0) return [];
  const parsed = items.map((it) => parseExplanationItem(it, type)).filter(Boolean);
  const maxAbs = Math.max(...parsed.map((p) => Math.abs(p.value)), 0.0001);

  return parsed.map((p) => {
    const ratio = Math.abs(p.value) / maxAbs; // 0.0 to 1.0
    const widthPct = Math.round(Math.max(ratio * 50, 4) * 10) / 10;
    return {
      ...p,
      ratio,
      widthPct,
      formattedValue: `${p.value >= 0 ? '+' : ''}${p.value.toFixed(4)}`,
      indicator: p.value >= 0 ? '↑' : '↓',
    };
  });
}

/**
 * Converts wizard form state into the final BRFSS payload dictionary.
 * Skipped or unselected fields are completely omitted from the payload object.
 */
export function buildBRFSSPayload(formData) {
  const payload = {};

  // Required: Age
  if (formData.age !== undefined && formData.age !== '' && formData.age !== null) {
    const band = mapAgeToBand(Number(formData.age));
    if (band !== null) payload.Age = band;
  }

  // Required: Sex
  if (formData.sex !== undefined && formData.sex !== '' && formData.sex !== null) {
    const s = Number(formData.sex);
    if (s === 0 || s === 1) payload.Sex = s;
  }

  // Required: Height & Weight -> BMI
  if (formData.height && formData.weight) {
    const bmi = computeBMI(formData.height, formData.weight);
    if (bmi !== null) payload.BMI = bmi;
  }

  // Required: General health
  if (formData.genHlth !== undefined && formData.genHlth !== '' && formData.genHlth !== null) {
    const gh = Number(formData.genHlth);
    if (gh >= 1 && gh <= 5) payload.GenHlth = gh;
  }

  // Optional: Binary fields
  for (const field of BINARY_FIELDS) {
    const val = formData[field.key];
    if (val === true || val === 1 || val === '1') {
      payload[field.key] = 1.0;
    } else if (val === false || val === 0 || val === '0') {
      payload[field.key] = 0.0;
    }
  }

  // Optional: Mental & Physical health days (0-30)
  if (formData.mentHlth !== undefined && formData.mentHlth !== '' && formData.mentHlth !== null) {
    const m = Number(formData.mentHlth);
    if (!isNaN(m) && m >= 0 && m <= 30) payload.MentHlth = m;
  }
  if (formData.physHlth !== undefined && formData.physHlth !== '' && formData.physHlth !== null) {
    const p = Number(formData.physHlth);
    if (!isNaN(p) && p >= 0 && p <= 30) payload.PhysHlth = p;
  }

  // Optional: Education (1-6)
  if (formData.education !== undefined && formData.education !== '' && formData.education !== null) {
    const ed = Number(formData.education);
    if (!isNaN(ed) && ed >= 1 && ed <= 6) payload.Education = ed;
  }

  // Optional: Income (1-8)
  if (formData.income !== undefined && formData.income !== '' && formData.income !== null) {
    const inc = Number(formData.income);
    if (!isNaN(inc) && inc >= 1 && inc <= 8) payload.Income = inc;
  }

  return payload;
}
