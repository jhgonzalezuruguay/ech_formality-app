import { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
} from 'recharts';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import modelParams from './ech_model_params.json';
import trainSummary from './ech_train_summary.json';
import lockedTest from './ech_locked_test.json';

// Direct imports of your exported ECH 2025 JSON files
// Place ech_model_params.json, ech_train_summary.json, and ech_locked_test.json
// in the same directory as App.tsx (e.g., src/).
import modelParams from './ech_model_params.json';
import trainSummary from './ech_train_summary.json';
import lockedTestData from './ech_locked_test.json';

interface CoefficientRow {
  term: string;
  estimate: number;
  std_error: number;
  odds_ratio: number;
}

interface CalibrationBin {
  bin: number;
  bin_label: string;
  n: number;
  mean_pred: number;
  observed_rate: number;
}

interface FirmSummaryRow {
  firm_size: string;
  n: number;
  observed_rate: number;
  mean_pred: number;
}

interface OccSummaryRow {
  occupation: string;
  n: number;
  observed_rate: number;
  mean_pred: number;
}

interface PovertySummaryRow {
  is_poor_income: number;
  is_poor_multi: number;
  n: number;
  observed_rate: number;
  mean_pred: number;
}

interface LockedTestRow {
  y: number;
  fs: string;
  occ: string;
  age: number;
  edu: number;
  hh: number;
  pi: number;
  pm: number;
}

interface TestEvaluationResult {
  nTest: number;
  observedFormalityRate: number;
  meanPredictedProb: number;
  testLog10Score: number;
  testAvgLog10Score: number;
  nullLog10Score: number;
  nullAvgLog10Score: number;
  scoreGainOverNull: number;
  calibrationBins: CalibrationBin[];
  byFirmSize: FirmSummaryRow[];
  byOccupation: OccSummaryRow[];
  byPoverty: {
    label: string;
    n: number;
    observed_rate: number;
    mean_pred: number;
  }[];
}

const FIRM_SIZE_ORDER = [
  '1_person',
  '2_to_4',
  '5_to_9',
  '10_to_49',
  '50_plus',
  'Public_Sector',
  'Not_Applicable_or_DK',
];

const FIRM_SIZE_LABELS: Record<string, string> = {
  '1_person': '1 Person (Solo)',
  '2_to_4': '2 to 4 Workers',
  '5_to_9': '5 to 9 Workers',
  '10_to_49': '10 to 49 Workers',
  '50_plus': '50+ Workers',
  Public_Sector: 'Public Sector',
  Not_Applicable_or_DK: 'N/A or Unknown',
};

const OCCUPATION_ORDER = [
  'Private Wage Worker',
  'Public Employee',
  'Self-Employed',
  'Other',
];

const POVERTY_ORDER = [
  'Neither Poor',
  'Multidim. Poor Only',
  'Income Poor Only',
  'Income + Multidim. Poor',
];

const TERM_LABELS: Record<string, string> = {
  '(Intercept)': 'Intercept (Baseline)',
  firm_size2_to_4: 'Firm Size: 2 to 4 (vs. 1 Person)',
  firm_size5_to_9: 'Firm Size: 5 to 9 (vs. 1 Person)',
  firm_size10_to_49: 'Firm Size: 10 to 49 (vs. 1 Person)',
  firm_size50_plus: 'Firm Size: 50+ (vs. 1 Person)',
  firm_sizePublic_Sector: 'Sector: Public Sector (vs. 1 Person)',
  firm_sizeNot_Applicable_or_DK: 'Firm Size: N/A or Unknown (vs. 1 Person)',
  'occ_for_modelSelf-Employed': 'Occupation: Self-Employed (vs. Private Wage)',
  occ_for_modelOther: 'Occupation: Other (vs. Private Wage)',
  age: 'Age (per year)',
  education: 'Education Level (per unit)',
  household_size: 'Household Size (per member)',
  is_poor_income: 'Income Poverty (Yes = 1)',
  is_poor_multi: 'Multidimensional Poverty (Yes = 1)',
};

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

function calcLog10Score(y: number[], p: number[]): number {
  const eps = 1e-12;
  let total = 0;
  for (let i = 0; i < y.length; i++) {
    const pClip = Math.min(Math.max(p[i], eps), 1 - eps);
    total += y[i] * Math.log10(pClip) + (1 - y[i]) * Math.log10(1 - pClip);
  }
  return total;
}

export default function App() {
  // Interactive Worker Probability Calculator State
  const [calcFirmSize, setCalcFirmSize] = useState<string>('1_person');
  const [calcOccupation, setCalcOccupation] = useState<string>('Private Wage Worker');
  const [calcAge, setCalcAge] = useState<number>(38);
  const [calcEducation, setCalcEducation] = useState<number>(3);
  const [calcHouseholdSize, setCalcHouseholdSize] = useState<number>(3);
  const [calcPoorIncome, setCalcPoorIncome] = useState<boolean>(false);
  const [calcPoorMulti, setCalcPoorMulti] = useState<boolean>(false);

  // Single-use Locked Test Set State
  const [confirmUnlockModal, setConfirmUnlockModal] = useState<boolean>(false);
  const [testUnlocked, setTestUnlocked] = useState<boolean>(false);
  const [testResults, setTestResults] = useState<TestEvaluationResult | null>(null);
  const [reflectionNotes, setReflectionNotes] = useState<string>(
    'Prior Hypothesis: Public sector and larger private firms (50+ workers) dominate formal registration in Uruguay, while income and multidimensional poverty sharply reduce formality odds.'
  );

  // Build coefficient lookup map from ech_model_params_2.json
  const coefMap = useMemo(() => {
    const map: Record<string, number> = {};
    (modelParams.coefficients as CoefficientRow[]).forEach((row) => {
      map[row.term] = row.estimate;
    });
    return map;
  }, []);

  // Predict P(is_formal = 1 | x) using exact trained logistic parameters
  const predictProbability = useMemo(() => {
    return (params: {
      fs: string;
      occ: string;
      age: number;
      edu: number;
      hh: number;
      pi: number;
      pm: number;
    }) => {
      let z = coefMap['(Intercept)'] ?? 0;

      if (params.fs !== '1_person') {
        const fsKey = `firm_size${params.fs}`;
        z += coefMap[fsKey] ?? 0;
      }

      // Match training coding: Public Employee uses Private Wage Worker baseline in occ_for_model
      const occModel =
        params.occ === 'Public Employee' ? 'Private Wage Worker' : params.occ;
      if (occModel !== 'Private Wage Worker') {
        const occKey = `occ_for_model${occModel}`;
        z += coefMap[occKey] ?? 0;
      }

      z += (coefMap['age'] ?? 0) * params.age;
      z += (coefMap['education'] ?? 0) * params.edu;
      z += (coefMap['household_size'] ?? 0) * params.hh;
      z += (coefMap['is_poor_income'] ?? 0) * params.pi;
      z += (coefMap['is_poor_multi'] ?? 0) * params.pm;

      return { logOdds: z, prob: sigmoid(z) };
    };
  }, [coefMap]);

  // Live calculator output
  const livePrediction = useMemo(() => {
    return predictProbability({
      fs: calcFirmSize,
      occ: calcOccupation,
      age: calcAge,
      edu: calcEducation,
      hh: calcHouseholdSize,
      pi: calcPoorIncome ? 1 : 0,
      pm: calcPoorMulti ? 1 : 0,
    });
  }, [
    predictProbability,
    calcFirmSize,
    calcOccupation,
    calcAge,
    calcEducation,
    calcHouseholdSize,
    calcPoorIncome,
    calcPoorMulti,
  ]);

  // Format training charts data
  const trainCalibrationData = useMemo(() => {
    return (trainSummary.calibration_bins as CalibrationBin[]).map((b) => ({
      ...b,
      predictedPct: Number((b.mean_pred * 100).toFixed(1)),
      observedPct: Number((b.observed_rate * 100).toFixed(1)),
    }));
  }, []);

  const trainFirmData = useMemo(() => {
    return (trainSummary.by_firm_size as FirmSummaryRow[]).map((f) => ({
      ...f,
      label: FIRM_SIZE_LABELS[f.firm_size] ?? f.firm_size,
      observedPct: Number((f.observed_rate * 100).toFixed(1)),
      predictedPct: Number((f.mean_pred * 100).toFixed(1)),
    }));
  }, []);

  const trainPovertyData = useMemo(() => {
    return (trainSummary.by_poverty as PovertySummaryRow[]).map((p) => {
      let label = 'Neither Poor';
      if (p.is_poor_income === 1 && p.is_poor_multi === 1) {
        label = 'Income + Multidim. Poor';
      } else if (p.is_poor_income === 1) {
        label = 'Income Poor Only';
      } else if (p.is_poor_multi === 1) {
        label = 'Multidim. Poor Only';
      }
      return {
        ...p,
        label,
        observedPct: Number((p.observed_rate * 100).toFixed(1)),
        predictedPct: Number((p.mean_pred * 100).toFixed(1)),
      };
    });
  }, []);

  // One-time unlock handler for ech_locked_test.json
  const handleUnlockLockedTest = () => {
    if (testUnlocked) return;

    const rows = lockedTestData as LockedTestRow[];
    const yArr: number[] = [];
    const pArr: number[] = [];

    const binBuckets: Record<
      number,
      { bin: number; bin_label: string; n: number; sumP: number; sumY: number }
    > = {};
    for (let b = 1; b <= 10; b++) {
      binBuckets[b] = {
        bin: b,
        bin_label: `${(b - 1) * 10}-${b * 10}%`,
        n: 0,
        sumP: 0,
        sumY: 0,
      };
    }

    const firmBuckets: Record<
      string,
      { firm_size: string; n: number; sumP: number; sumY: number }
    > = {};

    const occBuckets: Record<
      string,
      { occupation: string; n: number; sumP: number; sumY: number }
    > = {};

    const povBuckets: Record<
      string,
      { label: string; n: number; sumP: number; sumY: number }
    > = {};

    rows.forEach((row) => {
      const { prob } = predictProbability({
        fs: row.fs,
        occ: row.occ,
        age: row.age,
        edu: row.edu,
        hh: row.hh,
        pi: row.pi,
        pm: row.pm,
      });

      yArr.push(row.y);
      pArr.push(prob);

      const binIdx = Math.min(Math.floor(prob * 10) + 1, 10);
      binBuckets[binIdx].n += 1;
      binBuckets[binIdx].sumP += prob;
      binBuckets[binIdx].sumY += row.y;

      if (!firmBuckets[row.fs]) {
        firmBuckets[row.fs] = { firm_size: row.fs, n: 0, sumP: 0, sumY: 0 };
      }
      firmBuckets[row.fs].n += 1;
      firmBuckets[row.fs].sumP += prob;
      firmBuckets[row.fs].sumY += row.y;

      if (!occBuckets[row.occ]) {
        occBuckets[row.occ] = { occupation: row.occ, n: 0, sumP: 0, sumY: 0 };
      }
      occBuckets[row.occ].n += 1;
      occBuckets[row.occ].sumP += prob;
      occBuckets[row.occ].sumY += row.y;

      let povLabel = 'Neither Poor';
      if (row.pi === 1 && row.pm === 1) povLabel = 'Income + Multidim. Poor';
      else if (row.pi === 1) povLabel = 'Income Poor Only';
      else if (row.pm === 1) povLabel = 'Multidim. Poor Only';

      if (!povBuckets[povLabel]) {
        povBuckets[povLabel] = { label: povLabel, n: 0, sumP: 0, sumY: 0 };
      }
      povBuckets[povLabel].n += 1;
      povBuckets[povLabel].sumP += prob;
      povBuckets[povLabel].sumY += row.y;
    });

    const nTest = yArr.length;
    const observedFormalityRate = yArr.reduce((a, b) => a + b, 0) / nTest;
    const meanPredictedProb = pArr.reduce((a, b) => a + b, 0) / nTest;
    const testLog10Score = calcLog10Score(yArr, pArr);
    const testAvgLog10Score = testLog10Score / nTest;

    // Compare against training base rate baseline (no peeking at test labels)
    const trainBaseRate = modelParams.train_base_rate;
    const nullProbs = new Array(nTest).fill(trainBaseRate);
    const nullLog10Score = calcLog10Score(yArr, nullProbs);
    const nullAvgLog10Score = nullLog10Score / nTest;

    const calibrationBins: CalibrationBin[] = Object.values(binBuckets)
      .filter((b) => b.n > 0)
      .map((b) => ({
        bin: b.bin,
        bin_label: b.bin_label,
        n: b.n,
        mean_pred: Number((b.sumP / b.n).toFixed(4)),
        observed_rate: Number((b.sumY / b.n).toFixed(4)),
      }));

    const byFirmSize: FirmSummaryRow[] = Object.values(firmBuckets)
      .sort(
        (a, b) =>
          FIRM_SIZE_ORDER.indexOf(a.firm_size) - FIRM_SIZE_ORDER.indexOf(b.firm_size)
      )
      .map((f) => ({
        firm_size: f.firm_size,
        n: f.n,
        observed_rate: Number((f.sumY / f.n).toFixed(4)),
        mean_pred: Number((f.sumP / f.n).toFixed(4)),
      }));

    const byOccupation: OccSummaryRow[] = Object.values(occBuckets)
      .sort(
        (a, b) =>
          OCCUPATION_ORDER.indexOf(a.occupation) - OCCUPATION_ORDER.indexOf(b.occupation)
      )
      .map((o) => ({
        occupation: o.occupation,
        n: o.n,
        observed_rate: Number((o.sumY / o.n).toFixed(4)),
        mean_pred: Number((o.sumP / o.n).toFixed(4)),
      }));

    const byPoverty = Object.values(povBuckets)
      .sort((a, b) => POVERTY_ORDER.indexOf(a.label) - POVERTY_ORDER.indexOf(b.label))
      .map((p) => ({
        label: p.label,
        n: p.n,
        observed_rate: Number((p.sumY / p.n).toFixed(4)),
        mean_pred: Number((p.sumP / p.n).toFixed(4)),
      }));

    setTestResults({
      nTest,
      observedFormalityRate,
      meanPredictedProb,
      testLog10Score,
      testAvgLog10Score,
      nullLog10Score,
      nullAvgLog10Score,
      scoreGainOverNull: testLog10Score - nullLog10Score,
      calibrationBins,
      byFirmSize,
      byOccupation,
      byPoverty,
    });
    setTestUnlocked(true);
    setConfirmUnlockModal(false);
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f8fafc',
        color: '#0f172a',
        fontFamily:
          'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        fontSize: '16px',
        lineHeight: 1.6,
        padding: '24px 16px 64px',
      }}
    >
      <main style={{ maxWidth: '1120px', margin: '0 auto' }}>
        {/* Header */}
        <header
          style={{
            backgroundColor: '#0f172a',
            color: '#f8fafc',
            borderRadius: '14px',
            padding: '28px 32px',
            marginBottom: '28px',
            boxShadow: '0 4px 16px rgba(15, 23, 42, 0.12)',
          }}
        >
          <p
            style={{
              fontSize: '16px',
              fontWeight: 600,
              color: '#38bdf8',
              margin: '0 0 8px',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            Probability for AI &bull; Final Project &bull; Real Data ({modelParams.dataset_source})
          </p>
          <h1
            style={{
              fontSize: '30px',
              fontWeight: 800,
              margin: '0 0 12px',
              lineHeight: 1.25,
            }}
          >
            Predicting Worker Formality in Uruguay (ECH 2025)
          </h1>
          <p style={{ fontSize: '17px', color: '#e2e8f0', margin: 0 }}>
            Estimating <InlineMath math="P(	ext{is\_formal} = 1 \mid \mathbf{x})" /> across{' '}
            <strong>{modelParams.n_total.toLocaleString()}</strong> employed workers using a
            stratified 80% training set (<strong>{modelParams.n_train.toLocaleString()}</strong>) and
            a sealed 20% locked test set (<strong>{modelParams.n_locked_test.toLocaleString()}</strong>
            , seed = {modelParams.split_seed}).
          </p>
        </header>

        {/* Section 1: Generative Story & Probability Model */}
        <section
          aria-labelledby="model-story-heading"
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #cbd5e1',
            borderRadius: '12px',
            padding: '24px 28px',
            marginBottom: '28px',
          }}
        >
          <h2
            id="model-story-heading"
            style={{ fontSize: '22px', fontWeight: 700, marginTop: 0, marginBottom: '12px' }}
          >
            1. Generative Story &amp; Logistic Probability Model
          </h2>
          <p style={{ fontSize: '16px', marginTop: 0 }}>
            Each employed worker <InlineMath math="i" /> in Uruguay has a binary social-security
            registration outcome <InlineMath math="Y_i \in \{0, 1\}" /> (where{' '}
            <InlineMath math="Y_i = 1" /> means formal registration in BPS/pension coverage,{' '}
            <code>f82 == 1</code>). We model <InlineMath math="Y_i" /> as a Bernoulli random
            variable whose probability <InlineMath math="p_i" /> is determined on the log-odds
            scale by workplace structure, occupation, human capital, and household poverty:
          </p>
          <div
            style={{
              backgroundColor: '#f1f5f9',
              borderRadius: '8px',
              padding: '16px',
              margin: '16px 0',
              overflowX: 'auto',
            }}
          >
            <BlockMath
              math={
                'Y_i \sim \text{Bernoulli}(p_i), \qquad \log\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \mathbf{x}_{\text{firm},i}^{\top}\boldsymbol{\beta}_{\text{firm}} + \mathbf{x}_{\text{occ},i}^{\top}\boldsymbol{\beta}_{\text{occ}} + \beta_{\text{age}}\text{Age}_i + \beta_{\text{edu}}\text{Edu}_i + \beta_{\text{hh}}\text{HH}_i + \beta_{\text{pi}}\text{PoorInc}_i + \beta_{\text{pm}}\text{PoorMulti}_i'
              }
            />
          </div>

          {/* Training Score KPI Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
              marginTop: '20px',
            }}
          >
            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #cbd5e1',
                borderRadius: '10px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '16px', color: '#475569', fontWeight: 600 }}>
                Training Base Rate
              </div>
              <div style={{ fontSize: '26px', fontWeight: 800, color: '#0f172a' }}>
                {(modelParams.train_base_rate * 100).toFixed(2)}%
              </div>
              <div style={{ fontSize: '16px', color: '#475569' }}>
                n = {modelParams.n_train.toLocaleString()} workers
              </div>
            </div>

            <div
              style={{
                backgroundColor: '#eff6ff',
                border: '1px solid #93c5fd',
                borderRadius: '10px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '16px', color: '#1e40af', fontWeight: 600 }}>
                Training Score (<InlineMath math="\log_{10} L" />)
              </div>
              <div style={{ fontSize: '26px', fontWeight: 800, color: '#1d4ed8' }}>
                {modelParams.train_log10_score.toLocaleString()}
              </div>
              <div style={{ fontSize: '16px', color: '#1e3a8a' }}>
                Avg: {modelParams.train_avg_log10_score.toFixed(4)} / worker
              </div>
            </div>

            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #cbd5e1',
                borderRadius: '10px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '16px', color: '#475569', fontWeight: 600 }}>
                Baseline Null Score (<InlineMath math="\log_{10} L_0" />)
              </div>
              <div style={{ fontSize: '26px', fontWeight: 800, color: '#334155' }}>
                {modelParams.train_null_log10_score.toLocaleString()}
              </div>
              <div style={{ fontSize: '16px', color: '#475569' }}>
                Constant p = {(modelParams.train_base_rate * 100).toFixed(2)}%
              </div>
            </div>

            <div
              style={{
                backgroundColor: '#ecfdf5',
                border: '1px solid #6ee7b7',
                borderRadius: '10px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '16px', color: '#065f46', fontWeight: 600 }}>
                Training Score Gain
              </div>
              <div style={{ fontSize: '26px', fontWeight: 800, color: '#047857' }}>
                +{(modelParams.train_log10_score - modelParams.train_null_log10_score).toFixed(2)}
              </div>
              <div style={{ fontSize: '16px', color: '#065f46' }}>
                Higher <InlineMath math="\log_{10}" /> likelihood is better
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Interactive Worker Probability Calculator */}
        <section
          aria-labelledby="calculator-heading"
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #cbd5e1',
            borderRadius: '12px',
            padding: '24px 28px',
            marginBottom: '28px',
          }}
        >
          <h2
            id="calculator-heading"
            style={{ fontSize: '22px', fontWeight: 700, marginTop: 0, marginBottom: '12px' }}
          >
            2. Interactive Worker Formality Calculator (Trained Parameters)
          </h2>
          <p style={{ fontSize: '16px', marginTop: 0, marginBottom: '20px' }}>
            Adjust a worker&apos;s workplace and socioeconomic attributes to compute their predicted
            probability of formal employment <InlineMath math="P(	ext{is\_formal} = 1 \mid \mathbf{x})" />{' '}
            directly from the parameters fitted on the 80% training set.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '20px',
              alignItems: 'start',
            }}
          >
            {/* Controls Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label
                  htmlFor="calc-firm-size"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Firm Size / Sector
                </label>
                <select
                  id="calc-firm-size"
                  value={calcFirmSize}
                  onChange={(e) => setCalcFirmSize(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    fontSize: '16px',
                    borderRadius: '8px',
                    border: '1px solid #94a3b8',
                    backgroundColor: '#ffffff',
                  }}
                >
                  {Object.entries(FIRM_SIZE_LABELS).map(([val, label]) => (
                    <option key={val} value={val}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="calc-occupation"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Occupation Category
                </label>
                <select
                  id="calc-occupation"
                  value={calcOccupation}
                  onChange={(e) => setCalcOccupation(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    fontSize: '16px',
                    borderRadius: '8px',
                    border: '1px solid #94a3b8',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <option value="Private Wage Worker">Private Wage Worker</option>
                  <option value="Public Employee">Public Employee</option>
                  <option value="Self-Employed">Self-Employed</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="calc-age"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Age: <strong>{calcAge} years</strong>
                </label>
                <input
                  id="calc-age"
                  type="range"
                  min={14}
                  max={80}
                  value={calcAge}
                  onChange={(e) => setCalcAge(Number(e.target.value))}
                  style={{ width: '100%', accentColor: '#1d4ed8', cursor: 'pointer' }}
                />
              </div>

              <div>
                <label
                  htmlFor="calc-edu"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Education Level Code (<code>e51_2</code>): <strong>{calcEducation}</strong>
                </label>
                <input
                  id="calc-edu"
                  type="range"
                  min={0}
                  max={10}
                  value={calcEducation}
                  onChange={(e) => setCalcEducation(Number(e.target.value))}
                  style={{ width: '100%', accentColor: '#1d4ed8', cursor: 'pointer' }}
                />
              </div>

              <div>
                <label
                  htmlFor="calc-hh"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Household Size (<code>HT19</code>): <strong>{calcHouseholdSize} members</strong>
                </label>
                <input
                  id="calc-hh"
                  type="range"
                  min={1}
                  max={12}
                  value={calcHouseholdSize}
                  onChange={(e) => setCalcHouseholdSize(Number(e.target.value))}
                  style={{ width: '100%', accentColor: '#1d4ed8', cursor: 'pointer' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', paddingTop: '4px' }}>
                <label
                  htmlFor="calc-poor-inc"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '16px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  <input
                    id="calc-poor-inc"
                    type="checkbox"
                    checked={calcPoorIncome}
                    onChange={(e) => setCalcPoorIncome(e.target.checked)}
                    style={{ width: '20px', height: '20px', accentColor: '#1d4ed8' }}
                  />
                  Income Poverty (<code>pobre17 == 1</code>)
                </label>

                <label
                  htmlFor="calc-poor-multi"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '16px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  <input
                    id="calc-poor-multi"
                    type="checkbox"
                    checked={calcPoorMulti}
                    onChange={(e) => setCalcPoorMulti(e.target.checked)}
                    style={{ width: '20px', height: '20px', accentColor: '#1d4ed8' }}
                  />
                  Multidimensional Poverty (<code>pobre_multi == 1</code>)
                </label>
              </div>
            </div>

            {/* Live Probability Output Card */}
            <div
              role="region"
              aria-live="polite"
              style={{
                backgroundColor: '#0f172a',
                color: '#f8fafc',
                borderRadius: '12px',
                padding: '24px',
              }}
            >
              <div style={{ fontSize: '16px', color: '#94a3b8', fontWeight: 600 }}>
                Predicted Probability of Formal Employment
              </div>
              <div
                style={{
                  fontSize: '44px',
                  fontWeight: 800,
                  color: livePrediction.prob >= 0.5 ? '#34d399' : '#fbbf24',
                  margin: '8px 0',
                }}
              >
                {(livePrediction.prob * 100).toFixed(1)}%
              </div>
              <p style={{ fontSize: '16px', color: '#cbd5e1', margin: '0 0 16px' }}>
                Log-odds <InlineMath math="z" /> ={' '}
                <strong>{livePrediction.logOdds.toFixed(4)}</strong> &bull; Odds ={' '}
                <strong>{Math.exp(livePrediction.logOdds).toFixed(3)}</strong>
              </p>

              {/* Visual Probability Gauge */}
              <div
                style={{
                  width: '100%',
                  height: '18px',
                  backgroundColor: '#334155',
                  borderRadius: '999px',
                  overflow: 'hidden',
                  marginBottom: '16px',
                }}
              >
                <div
                  style={{
                    width: `${Math.min(100, Math.max(0, livePrediction.prob * 100))}%`,
                    height: '100%',
                    backgroundColor: livePrediction.prob >= 0.5 ? '#10b981' : '#f59e0b',
                    transition: 'width 0.2s ease',
                  }}
                />
              </div>

              <div
                style={{
                  backgroundColor: '#1e293b',
                  borderRadius: '8px',
                  padding: '14px',
                  fontSize: '16px',
                  color: '#e2e8f0',
                }}
              >
                <strong>Interpretation:</strong> Among 100 Uruguayan workers with this exact
                profile, our model expects roughly{' '}
                <strong>{Math.round(livePrediction.prob * 100)}</strong> to hold formal registration
                and <strong>{100 - Math.round(livePrediction.prob * 100)}</strong> to work
                informally.
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Training-Set Calibration & Subgroup Patterns */}
        <section
          aria-labelledby="training-charts-heading"
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #cbd5e1',
            borderRadius: '12px',
            padding: '24px 28px',
            marginBottom: '28px',
          }}
        >
          <h2
            id="training-charts-heading"
            style={{ fontSize: '22px', fontWeight: 700, marginTop: 0, marginBottom: '12px' }}
          >
            3. Training-Set Calibration &amp; Subgroup Patterns (n = {modelParams.n_train.toLocaleString()})
          </h2>
          <p style={{ fontSize: '16px', marginTop: 0, marginBottom: '24px' }}>
            Before unlocking the test set, we verify whether predicted probabilities on the 80%
            training set are <strong>calibrated</strong>, <strong>overconfident</strong>, or{' '}
            <strong>underconfident</strong> across decile bins and key socioeconomic subgroups.
          </p>

          {/* Chart A: 10-Bin Calibration Curve */}
          <div style={{ marginBottom: '36px' }}>
            <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
              A. Training Calibration by Predicted Probability Decile (%)
            </h3>
            <div style={{ width: '100%', height: '340px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={trainCalibrationData}
                  margin={{ top: 16, right: 24, left: 16, bottom: 24 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                  <XAxis
                    dataKey="bin_label"
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                    label={{
                      value: 'Predicted Formality Probability Bin',
                      position: 'insideBottom',
                      offset: -14,
                      fontSize: 16,
                      fill: '#0f172a',
                    }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                    label={{
                      value: 'Formality Rate (%)',
                      angle: -90,
                      position: 'insideLeft',
                      fontSize: 16,
                      fill: '#0f172a',
                    }}
                  />
                  <Tooltip contentStyle={{ fontSize: '16px' }} />
                  <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '12px' }} />
                  <Line
                    type="monotone"
                    dataKey="predictedPct"
                    name="Mean Predicted (%)"
                    stroke="#2563eb"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="observedPct"
                    name="Observed Formal (%)"
                    stroke="#059669"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart B: Formality by Firm Size */}
          <div style={{ marginBottom: '36px' }}>
            <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
              B. Training Formality Rate by Firm Size &amp; Sector (%)
            </h3>
            <div style={{ width: '100%', height: '360px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={trainFirmData}
                  margin={{ top: 16, right: 24, left: 16, bottom: 48 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                  <XAxis
                    dataKey="label"
                    interval={0}
                    angle={-18}
                    textAnchor="end"
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                    label={{
                      value: 'Formal Workers (%)',
                      angle: -90,
                      position: 'insideLeft',
                      fontSize: 16,
                      fill: '#0f172a',
                    }}
                  />
                  <Tooltip contentStyle={{ fontSize: '16px' }} />
                  <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '24px' }} />
                  <Bar
                    dataKey="observedPct"
                    name="Observed Formal (%)"
                    fill="#0f766e"
                    radius={[6, 6, 0, 0]}
                  />
                  <Bar
                    dataKey="predictedPct"
                    name="Mean Predicted (%)"
                    fill="#3b82f6"
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart C: Formality by Poverty Subgroup */}
          <div>
            <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
              C. Training Formality Rate by Poverty Status (%)
            </h3>
            <div style={{ width: '100%', height: '340px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={trainPovertyData}
                  margin={{ top: 16, right: 24, left: 16, bottom: 28 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                  <XAxis
                    dataKey="label"
                    interval={0}
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 16, fill: '#0f172a' }}
                    label={{
                      value: 'Formal Workers (%)',
                      angle: -90,
                      position: 'insideLeft',
                      fontSize: 16,
                      fill: '#0f172a',
                    }}
                  />
                  <Tooltip contentStyle={{ fontSize: '16px' }} />
                  <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '12px' }} />
                  <Bar
                    dataKey="observedPct"
                    name="Observed Formal (%)"
                    fill="#7c3aed"
                    radius={[6, 6, 0, 0]}
                  />
                  <Bar
                    dataKey="predictedPct"
                    name="Mean Predicted (%)"
                    fill="#38bdf8"
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* Section 4: Fitted Parameter Table */}
        <section
          aria-labelledby="params-table-heading"
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #cbd5e1',
            borderRadius: '12px',
            padding: '24px 28px',
            marginBottom: '28px',
            overflowX: 'auto',
          }}
        >
          <h2
            id="params-table-heading"
            style={{ fontSize: '22px', fontWeight: 700, marginTop: 0, marginBottom: '12px' }}
          >
            4. Fitted Logistic Model Parameters (Training Set Only)
          </h2>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '16px',
              textAlign: 'left',
            }}
          >
            <thead>
              <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '2px solid #94a3b8' }}>
                <th style={{ padding: '12px' }}>Feature / Parameter</th>
                <th style={{ padding: '12px' }}>
                  Log-Odds (<InlineMath math="\hat{eta}" />)
                </th>
                <th style={{ padding: '12px' }}>Std. Error</th>
                <th style={{ padding: '12px' }}>
                  Odds Ratio (<InlineMath math="e^{\hat{eta}}" />)
                </th>
              </tr>
            </thead>
            <tbody>
              {(modelParams.coefficients as CoefficientRow[]).map((row) => (
                <tr key={row.term} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '12px', fontWeight: 600 }}>
                    {TERM_LABELS[row.term] ?? row.term}
                  </td>
                  <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '16px' }}>
                    {row.estimate >= 0 ? `+${row.estimate.toFixed(4)}` : row.estimate.toFixed(4)}
                  </td>
                  <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '16px' }}>
                    {row.std_error.toFixed(4)}
                  </td>
                  <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '16px' }}>
                    {row.odds_ratio.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* Section 5: Sealed Locked Test Set (Single-Use Unlock) */}
        <section
          aria-labelledby="locked-test-heading"
          style={{
            backgroundColor: testUnlocked ? '#ffffff' : '#fffbeb',
            border: testUnlocked ? '2px solid #059669' : '2px solid #f59e0b',
            borderRadius: '12px',
            padding: '24px 28px',
          }}
        >
          <h2
            id="locked-test-heading"
            style={{ fontSize: '22px', fontWeight: 700, marginTop: 0, marginBottom: '12px' }}
          >
            5. Locked Test Set Evaluation (n = {modelParams.n_locked_test.toLocaleString()} Unseen Workers)
          </h2>

          {!testUnlocked ? (
            <div>
              <p style={{ fontSize: '16px', marginTop: 0 }}>
                <strong>Status: SEALED.</strong> The 20% stratified test set (
                <code>ech_locked_test.json</code>) has never been used to fit or tune any
                parameter. To preserve an honest one-time evaluation, you may unlock it{' '}
                <strong>only once</strong> after freezing your model and recording your prior
                hypothesis below.
              </p>

              <div style={{ marginBottom: '18px' }}>
                <label
                  htmlFor="prior-hypothesis-notes"
                  style={{ display: 'block', fontWeight: 600, fontSize: '16px', marginBottom: '6px' }}
                >
                  Pre-Unlock Hypothesis &amp; Notes (Frozen Before Opening Test Set)
                </label>
                <textarea
                  id="prior-hypothesis-notes"
                  rows={3}
                  value={reflectionNotes}
                  onChange={(e) => setReflectionNotes(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    fontSize: '16px',
                    borderRadius: '8px',
                    border: '1px solid #94a3b8',
                    fontFamily: 'inherit',
                  }}
                />
              </div>

              {!confirmUnlockModal ? (
                <button
                  type="button"
                  onClick={() => setConfirmUnlockModal(true)}
                  style={{
                    backgroundColor: '#b45309',
                    color: '#ffffff',
                    fontSize: '17px',
                    fontWeight: 700,
                    padding: '14px 24px',
                    borderRadius: '10px',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                >
                  Unlock &amp; Score Locked Test Set Once
                </button>
              ) : (
                <div
                  role="alertdialog"
                  aria-labelledby="confirm-unlock-title"
                  style={{
                    backgroundColor: '#fef3c7',
                    border: '1px solid #d97706',
                    borderRadius: '10px',
                    padding: '18px',
                  }}
                >
                  <p
                    id="confirm-unlock-title"
                    style={{ fontSize: '17px', fontWeight: 700, margin: '0 0 8px' }}
                  >
                    Confirm One-Time Test Set Unlock
                  </p>
                  <p style={{ fontSize: '16px', margin: '0 0 14px' }}>
                    Once unlocked, no parameters may be changed or retrained. Are you ready to
                    evaluate your frozen model on all {modelParams.n_locked_test.toLocaleString()}{' '}
                    locked test workers?
                  </p>
                  <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={handleUnlockLockedTest}
                      style={{
                        backgroundColor: '#047857',
                        color: '#ffffff',
                        fontSize: '16px',
                        fontWeight: 700,
                        padding: '12px 20px',
                        borderRadius: '8px',
                        border: 'none',
                        cursor: 'pointer',
                      }}
                    >
                      Yes, Evaluate Locked Test Set Now (Permanent)
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmUnlockModal(false)}
                      style={{
                        backgroundColor: '#e2e8f0',
                        color: '#0f172a',
                        fontSize: '16px',
                        fontWeight: 600,
                        padding: '12px 20px',
                        borderRadius: '8px',
                        border: '1px solid #94a3b8',
                        cursor: 'pointer',
                      }}
                    >
                      Cancel (Keep Sealed)
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            testResults && (
              <div>
                <div
                  style={{
                    backgroundColor: '#ecfdf5',
                    border: '1px solid #6ee7b7',
                    borderRadius: '8px',
                    padding: '14px 18px',
                    marginBottom: '20px',
                    fontSize: '16px',
                    color: '#065f46',
                  }}
                >
                  <strong>UNLOCKED (ONE-TIME EVALUATION COMPLETE):</strong> Model parameters are
                  permanently frozen. Below are your out-of-sample results on{' '}
                  <strong>{testResults.nTest.toLocaleString()}</strong> unseen ECH 2025 workers.
                </div>

                {/* Test Score KPI Cards */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                    gap: '16px',
                    marginBottom: '28px',
                  }}
                >
                  <div
                    style={{
                      backgroundColor: '#eff6ff',
                      border: '1px solid #93c5fd',
                      borderRadius: '10px',
                      padding: '16px',
                    }}
                  >
                    <div style={{ fontSize: '16px', color: '#1e40af', fontWeight: 600 }}>
                      Locked Test Score (<InlineMath math="\log_{10} L" />)
                    </div>
                    <div style={{ fontSize: '26px', fontWeight: 800, color: '#1d4ed8' }}>
                      {testResults.testLog10Score.toFixed(2)}
                    </div>
                    <div style={{ fontSize: '16px', color: '#1e3a8a' }}>
                      Avg: {testResults.testAvgLog10Score.toFixed(4)} / worker
                    </div>
                  </div>

                  <div
                    style={{
                      backgroundColor: '#f8fafc',
                      border: '1px solid #cbd5e1',
                      borderRadius: '10px',
                      padding: '16px',
                    }}
                  >
                    <div style={{ fontSize: '16px', color: '#475569', fontWeight: 600 }}>
                      Test Baseline Score (<InlineMath math="\log_{10} L_0" />)
                    </div>
                    <div style={{ fontSize: '26px', fontWeight: 800, color: '#334155' }}>
                      {testResults.nullLog10Score.toFixed(2)}
                    </div>
                    <div style={{ fontSize: '16px', color: '#475569' }}>
                      Avg: {testResults.nullAvgLog10Score.toFixed(4)} / worker
                    </div>
                  </div>

                  <div
                    style={{
                      backgroundColor: '#ecfdf5',
                      border: '1px solid #6ee7b7',
                      borderRadius: '10px',
                      padding: '16px',
                    }}
                  >
                    <div style={{ fontSize: '16px', color: '#065f46', fontWeight: 600 }}>
                      Out-of-Sample Score Gain
                    </div>
                    <div style={{ fontSize: '26px', fontWeight: 800, color: '#047857' }}>
                      +{testResults.scoreGainOverNull.toFixed(2)}
                    </div>
                    <div style={{ fontSize: '16px', color: '#065f46' }}>
                      Beats baseline on unseen data
                    </div>
                  </div>

                  <div
                    style={{
                      backgroundColor: '#f8fafc',
                      border: '1px solid #cbd5e1',
                      borderRadius: '10px',
                      padding: '16px',
                    }}
                  >
                    <div style={{ fontSize: '16px', color: '#475569', fontWeight: 600 }}>
                      Observed vs. Predicted Rate
                    </div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>
                      {(testResults.observedFormalityRate * 100).toFixed(2)}% vs.{' '}
                      {(testResults.meanPredictedProb * 100).toFixed(2)}%
                    </div>
                    <div style={{ fontSize: '16px', color: '#475569' }}>
                      Overall test-set calibration
                    </div>
                  </div>
                </div>

                {/* Locked Test Chart 1: Calibration by Decile */}
                <div style={{ marginBottom: '36px' }}>
                  <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
                    A. Locked Test Calibration by Probability Decile (%)
                  </h3>
                  <div style={{ width: '100%', height: '340px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={testResults.calibrationBins.map((b) => ({
                          ...b,
                          predictedPct: Number((b.mean_pred * 100).toFixed(1)),
                          observedPct: Number((b.observed_rate * 100).toFixed(1)),
                        }))}
                        margin={{ top: 16, right: 24, left: 16, bottom: 24 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                        <XAxis
                          dataKey="bin_label"
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                          label={{
                            value: 'Predicted Formality Probability Bin (Locked Test)',
                            position: 'insideBottom',
                            offset: -14,
                            fontSize: 16,
                            fill: '#0f172a',
                          }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                          label={{
                            value: 'Formality Rate (%)',
                            angle: -90,
                            position: 'insideLeft',
                            fontSize: 16,
                            fill: '#0f172a',
                          }}
                        />
                        <Tooltip contentStyle={{ fontSize: '16px' }} />
                        <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '12px' }} />
                        <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" />
                        <Line
                          type="monotone"
                          dataKey="predictedPct"
                          name="Mean Predicted (%)"
                          stroke="#2563eb"
                          strokeWidth={3}
                          dot={{ r: 5 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="observedPct"
                          name="Observed Formal (%)"
                          stroke="#059669"
                          strokeWidth={3}
                          dot={{ r: 5 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Locked Test Chart 2: Formality by Firm Size & Sector */}
                <div style={{ marginBottom: '36px' }}>
                  <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
                    B. Locked Test Formality Rate by Firm Size &amp; Sector (%)
                  </h3>
                  <div style={{ width: '100%', height: '360px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={testResults.byFirmSize.map((f) => ({
                          ...f,
                          label: FIRM_SIZE_LABELS[f.firm_size] ?? f.firm_size,
                          observedPct: Number((f.observed_rate * 100).toFixed(1)),
                          predictedPct: Number((f.mean_pred * 100).toFixed(1)),
                        }))}
                        margin={{ top: 16, right: 24, left: 16, bottom: 48 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                        <XAxis
                          dataKey="label"
                          interval={0}
                          angle={-18}
                          textAnchor="end"
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                          label={{
                            value: 'Formal Workers (%)',
                            angle: -90,
                            position: 'insideLeft',
                            fontSize: 16,
                            fill: '#0f172a',
                          }}
                        />
                        <Tooltip contentStyle={{ fontSize: '16px' }} />
                        <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '24px' }} />
                        <Bar
                          dataKey="observedPct"
                          name="Observed Formal (%)"
                          fill="#0f766e"
                          radius={[6, 6, 0, 0]}
                        />
                        <Bar
                          dataKey="predictedPct"
                          name="Mean Predicted (%)"
                          fill="#3b82f6"
                          radius={[6, 6, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Locked Test Chart 3: Formality by Occupation Category */}
                <div style={{ marginBottom: '36px' }}>
                  <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
                    C. Locked Test Formality Rate by Occupation Category (%)
                  </h3>
                  <div style={{ width: '100%', height: '340px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={testResults.byOccupation.map((o) => ({
                          ...o,
                          observedPct: Number((o.observed_rate * 100).toFixed(1)),
                          predictedPct: Number((o.mean_pred * 100).toFixed(1)),
                        }))}
                        margin={{ top: 16, right: 24, left: 16, bottom: 28 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                        <XAxis
                          dataKey="occupation"
                          interval={0}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                          label={{
                            value: 'Formal Workers (%)',
                            angle: -90,
                            position: 'insideLeft',
                            fontSize: 16,
                            fill: '#0f172a',
                          }}
                        />
                        <Tooltip contentStyle={{ fontSize: '16px' }} />
                        <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '12px' }} />
                        <Bar
                          dataKey="observedPct"
                          name="Observed Formal (%)"
                          fill="#0284c7"
                          radius={[6, 6, 0, 0]}
                        />
                        <Bar
                          dataKey="predictedPct"
                          name="Mean Predicted (%)"
                          fill="#10b981"
                          radius={[6, 6, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Locked Test Chart 4: Formality by Poverty Status */}
                <div style={{ marginBottom: '24px' }}>
                  <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '8px' }}>
                    D. Locked Test Formality Rate by Poverty Status (%)
                  </h3>
                  <div style={{ width: '100%', height: '340px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={testResults.byPoverty.map((p) => ({
                          ...p,
                          observedPct: Number((p.observed_rate * 100).toFixed(1)),
                          predictedPct: Number((p.mean_pred * 100).toFixed(1)),
                        }))}
                        margin={{ top: 16, right: 24, left: 16, bottom: 28 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                        <XAxis
                          dataKey="label"
                          interval={0}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 16, fill: '#0f172a' }}
                          label={{
                            value: 'Formal Workers (%)',
                            angle: -90,
                            position: 'insideLeft',
                            fontSize: 16,
                            fill: '#0f172a',
                          }}
                        />
                        <Tooltip contentStyle={{ fontSize: '16px' }} />
                        <Legend wrapperStyle={{ fontSize: '16px', paddingTop: '12px' }} />
                        <Bar
                          dataKey="observedPct"
                          name="Observed Formal (%)"
                          fill="#7c3aed"
                          radius={[6, 6, 0, 0]}
                        />
                        <Bar
                          dataKey="predictedPct"
                          name="Mean Predicted (%)"
                          fill="#38bdf8"
                          radius={[6, 6, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Frozen Pre-Unlock Hypothesis Box */}
                <div
                  style={{
                    backgroundColor: '#f8fafc',
                    border: '1px solid #cbd5e1',
                    borderRadius: '8px',
                    padding: '16px',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '16px', marginBottom: '4px' }}>
                    Your Pre-Unlock Hypothesis (Recorded Before Opening Test Set):
                  </div>
                  <p style={{ margin: 0, fontSize: '16px', color: '#334155' }}>{reflectionNotes}</p>
                </div>
              </div>
            )
          )}
        </section>
      </main>
    </div>
  );
}
