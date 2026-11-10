import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  ArrowDown,
  ArrowRight,
  BadgeCheck,
  BookOpen,
  Briefcase,
  Check,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Code2,
  FileText,
  GraduationCap,
  Layers3,
  Lightbulb,
  LoaderCircle,
  MapPin,
  Plus,
  Rocket,
  Sparkles,
  Target,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  ChangeEvent,
  DragEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

interface ResumeEducation {
  degree: string;
  institution: string;
  year: string;
  cgpa: string;
}

interface ResumeExperience {
  role: string;
  company: string;
  duration: string;
  description: string[];
  tech: string[];
}

interface ResumeProject {
  name: string;
  description: string;
  tech: string[];
}

interface Resume {
  skills: string[];
  education: ResumeEducation[];
  experience: ResumeExperience[];
  projects: ResumeProject[];
}

interface Internship {
  job_id: string;
  title: string;
  company: string;
  location: string;
  work_mode: string;
  duration: string;
  eligibility: string;
  responsibilities: string[];
  required_skills: string[];
  preferred_skills: string[];
  description: string;
}

interface MatchScore {
  overall_score: number;
  semantic_similarity: number;
  required_coverage: number;
  preferred_coverage: number;
  matched_required: string[];
  matched_preferred: string[];
  missing_required: string[];
  missing_preferred: string[];
  required_match_count: number;
  preferred_match_count: number;
  summary: string;
}

interface InternshipMatch {
  internship: Internship;
  match_score: MatchScore;
}

interface LearningPathItem {
  topic: string;
  resource_type: string;
  estimated_hours: number;
}

interface GapAnalysis {
  missing_skills: string[];
  resume_critique: string;
  learning_path: LearningPathItem[];
}

interface MatchResponse {
  resume: Resume;
  top_internships: InternshipMatch[];
  gap_analysis: GapAnalysis | null;
}

const API_URL = "http://127.0.0.1:8000/api/v1/match";

const pipelineSteps = [
  "Reading resume & extracting structured candidate data...",
  "Generating embeddings & searching Qdrant vector database...",
  "Generating recruiter gap analysis with Gemini AI...",
];

const reveal = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.2, ease: "easeOut" } },
};

function isMatchResponse(value: unknown): value is MatchResponse {
  if (typeof value !== "object" || value === null) return false;
  const result = value as Partial<MatchResponse>;
  return (
    typeof result.resume === "object" &&
    result.resume !== null &&
    Array.isArray(result.top_internships) &&
    (result.gap_analysis === null ||
      (typeof result.gap_analysis === "object" && result.gap_analysis !== null))
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getErrorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = payload.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          typeof item === "object" && item !== null && "msg" in item
            ? String(item.msg)
            : String(item),
        )
        .join(", ");
    }
  }
  return fallback;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState<MatchResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isLoading) return;
    const timers = [
      window.setTimeout(() => setActiveStep(1), 4500),
      window.setTimeout(() => setActiveStep(2), 11000),
    ];
    return () => timers.forEach(window.clearTimeout);
  }, [isLoading]);

  const acceptFile = useCallback((candidate: File | undefined) => {
    if (!candidate) return;
    if (!/\.(pdf|docx|txt)$/i.test(candidate.name)) {
      setError("Unsupported file type. Please choose a .pdf, .docx, or .txt file.");
      return;
    }
    setError("");
    setResult(null);
    setFile(candidate);
  }, []);

  const onFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  };

  const analyzeResume = async () => {
    if (!file || isLoading) return;
    setError("");
    setResult(null);
    setActiveStep(0);
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, { method: "POST", body: formData });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(
          getErrorMessage(payload, `The matching service returned an error (${response.status}).`),
        );
      }
      if (!isMatchResponse(payload)) {
        throw new Error("The matching service returned an unexpected response. Please try again.");
      }
      setResult(payload);
      window.setTimeout(
        () => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "We couldn’t reach the matching service. Check that the API is running at 127.0.0.1:8000 and try again."
          : requestError instanceof Error
            ? requestError.message
            : "Something went wrong while analyzing your resume.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell min-h-screen bg-ink text-slate-900">
      <header className="relative z-10 border-b border-slate-200 bg-white">
        <div className="mx-auto flex h-[72px] max-w-[1440px] items-center justify-between px-5 sm:px-8">
          <a href="#" className="flex items-center gap-3" aria-label="Orbit home">
            <span className="brand-mark">
              <span />
              <span />
              <span />
            </span>
            <span className="text-[17px] font-semibold tracking-[-0.04em] text-slate-900">
              orbit<span className="text-violet">.</span>
            </span>
            <span className="hidden border-l border-slate-200 pl-3 text-xs text-slate-500 sm:block">
              Career intelligence
            </span>
          </a>
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5">
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
            <span className="text-[11px] font-medium tracking-wide text-slate-600">
              AI MATCHING <span className="text-emerald-700">ONLINE</span>
            </span>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-[1440px] px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
        <motion.section
          initial="hidden"
          animate="visible"
          variants={reveal}
          className="mb-9 grid items-end gap-7 lg:grid-cols-[1fr_auto]"
        >
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1.5">
              <Sparkles size={13} className="text-indigo-600" />
              <span className="text-[11px] font-semibold uppercase tracking-[0.13em] text-indigo-700">
                Your next opportunity starts here
              </span>
            </div>
            <h1 className="max-w-3xl text-[clamp(2.35rem,5vw,4.2rem)] font-semibold leading-[1.06] tracking-[-0.065em] text-slate-900">
              Make your next move
              <br className="hidden sm:block" /> with <span className="text-indigo-600">clarity.</span>
            </h1>
            <p className="mt-4 max-w-xl text-sm leading-6 text-slate-400 sm:text-[15px]">
              Upload your resume. Discover internships that fit, understand your gaps, and get a
              roadmap to close them.
            </p>
          </div>
          <div className="hidden items-center gap-4 pb-1 lg:flex">
            <div className="h-10 w-px bg-slate-200" />
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white">
                <BadgeCheck size={16} className="text-indigo-600" />
              </span>
              Private by design
              <CircleHelp size={13} className="text-slate-600" />
            </div>
          </div>
        </motion.section>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,0.65fr)]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.04, duration: 0.2 }}
            className="panel relative overflow-hidden p-5 sm:p-7"
          >
            <div className="relative">
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-sm font-medium text-white">
                    <span className="step-number">01</span>
                    Upload your resume
                  </div>
                  <p className="ml-9 mt-1.5 text-xs text-slate-500">
                    We’ll find the opportunities made for you.
                  </p>
                </div>
                <span className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 text-[10px] font-medium uppercase tracking-[0.14em] text-slate-500">
                  PDF, DOCX, TXT
                </span>
              </div>

              <input
                ref={inputRef}
                type="file"
                accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                className="sr-only"
                onChange={onFileInput}
                aria-label="Choose a PDF, DOCX, or TXT resume"
              />
              <motion.div
                onDragEnter={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                    setIsDragging(false);
                  }
                }}
                onDrop={onDrop}
                onClick={() => !file && inputRef.current?.click()}
                animate={{
                  scale: isDragging ? 1.012 : 1,
                  borderColor: isDragging ? "#4f46e5" : "#cbd5e1",
                  backgroundColor: isDragging ? "#eef2ff" : "#ffffff",
                }}
                transition={{ duration: 0.18 }}
                className={`upload-zone ${file ? "cursor-default" : "cursor-pointer"}`}
                role={file ? undefined : "button"}
                tabIndex={file ? undefined : 0}
                onKeyDown={(event) => {
                  if (!file && (event.key === "Enter" || event.key === " ")) {
                    event.preventDefault();
                    inputRef.current?.click();
                  }
                }}
                aria-label={file ? `Selected file ${file.name}` : "Drop a PDF, DOCX, or TXT here or browse"}
              >
                <AnimatePresence mode="wait">
                  {file ? (
                    <motion.div
                      key="file"
                      initial={{ opacity: 0, scale: 0.96, y: 6 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                      exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.2 } }}
                      className="flex w-full items-center gap-4"
                    >
                      <span className="file-icon-wrap">
                        <FileText size={23} strokeWidth={1.7} />
                        <span className="file-extension">{file.name.split(".").pop()?.toUpperCase()}</span>
                      </span>
                      <div className="min-w-0 flex-1 text-left">
                        <div className="truncate text-sm font-medium text-white">{file.name}</div>
                        <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                          <span>{formatBytes(file.size)}</span>
                          <span className="h-1 w-1 rounded-full bg-slate-700" />
                          <span className="flex items-center gap-1 text-emerald-700">
                            <CheckCircle2 size={12} /> Ready to analyze
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setFile(null);
                          setError("");
                          setResult(null);
                        }}
                        className="remove-file"
                        aria-label="Remove file"
                      >
                        <Trash2 size={15} />
                        <span className="hidden sm:inline">Remove</span>
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0, scale: 0.98 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, transition: { duration: 0.2 } }}
                      transition={{ duration: 0.2 }}
                      className="flex flex-col items-center text-center"
                    >
                      <span className="upload-icon">
                        <UploadCloud size={23} strokeWidth={1.7} />
                      </span>
                      <span className="mt-4 text-sm font-medium text-slate-200">
                        Drop your resume here
                      </span>
                      <span className="mt-1.5 text-xs text-slate-500">
                        or <span className="text-indigo-600">browse files</span> from your device
                      </span>
                      <span className="mt-4 text-[10px] tracking-wide text-slate-600">
                        PDF, DOCX, or TXT · Your file is only used for this analysis
                      </span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                <motion.button
                  type="button"
                  whileHover={file && !isLoading ? { scale: 1.01, y: -1 } : undefined}
                  whileTap={file && !isLoading ? { scale: 0.985 } : undefined}
                  disabled={!file || isLoading}
                  onClick={analyzeResume}
                  transition={{ duration: 0.15 }}
                  className="primary-button group flex-1"
                >
                  {isLoading ? (
                    <>
                      <LoaderCircle size={16} className="animate-spin" />
                      Analyzing your resume
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Analyze my resume
                      <ArrowRight
                        size={15}
                        className="ml-auto transition-transform group-hover:translate-x-1"
                      />
                    </>
                  )}
                </motion.button>
                <div className="flex items-center justify-center gap-2 rounded-xl border border-white/[0.07] bg-white/[0.025] px-4 py-3 text-[11px] text-slate-500">
                  <Layers3 size={14} className="text-slate-400" />
                  <span>3-step AI analysis</span>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.aside
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.2 }}
            className="panel relative flex min-h-[250px] flex-col justify-between overflow-hidden p-5 sm:p-7"
          >
            <div className="relative">
              <div className="flex items-center gap-2 text-sm font-medium text-white">
                <span className="step-number step-number-mint">02</span>
                Your match, in focus
              </div>
              <p className="ml-9 mt-1.5 text-xs leading-5 text-slate-500">
                More than a score. A clearer picture of what’s next.
              </p>
              <div className="preview-match mt-6">
                <div className="flex items-center gap-3">
                  <div className="company-avatar">
                    <Briefcase size={16} />
                  </div>
                  <div className="flex-1">
                    <div className="h-2 w-28 rounded bg-slate-300/20" />
                    <div className="mt-2 h-1.5 w-20 rounded bg-slate-500/20" />
                  </div>
                  <span className="match-ring">
                    <span>92</span>
                    <span>%</span>
                  </span>
                </div>
                <div className="mt-5 flex gap-1.5">
                  <span className="preview-chip w-[66px]" />
                  <span className="preview-chip w-[53px]" />
                  <span className="preview-chip w-[78px]" />
                  <span className="preview-chip w-[39px]" />
                </div>
              </div>
            </div>
            <div className="relative mt-5 flex items-center justify-between border-t border-white/[0.07] pt-4 text-[11px] text-slate-500">
              <span className="flex items-center gap-1.5">
                <Target size={13} className="text-emerald-300" />
                Skills · roles · growth
              </span>
              <span className="flex items-center gap-1 text-slate-400">
                Powered by AI <Sparkles size={11} className="text-violet-300" />
              </span>
            </div>
          </motion.aside>
        </section>

        <AnimatePresence>
          {isLoading && (
            <motion.section
              initial={{ opacity: 0, height: 0, y: -8 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: -8, transition: { duration: 0.2 } }}
              transition={{ duration: 0.2 }}
              className="mt-5 overflow-hidden"
              aria-live="polite"
            >
              <div className="panel relative overflow-hidden p-5 sm:p-6">
                <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
                  <div className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-violet-300/20 bg-violet-400/[0.08]">
                    <LoaderCircle size={21} className="animate-spin text-violet-200" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium text-white">Reading between the lines</p>
                        <p className="mt-1 text-xs text-slate-500">
                          Your analysis is underway. This can take a little while.
                        </p>
                      </div>
                      <span className="rounded-md border border-violet-300/15 bg-violet-300/[0.06] px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-violet-200">
                        Step {activeStep + 1} of 3
                      </span>
                    </div>
                    <div className="pipeline-track">
                      <motion.div
                        className="pipeline-progress"
                        animate={{ width: `${((activeStep + 1) / 3) * 100}%` }}
                        transition={{ duration: 0.2, ease: "easeInOut" }}
                      />
                    </div>
                    <div className="mt-3 grid gap-2 sm:grid-cols-3">
                      {pipelineSteps.map((step, index) => (
                        <div key={step} className="flex items-start gap-2 text-[11px] leading-4">
                          <span
                            className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${
                              index < activeStep
                                ? "bg-emerald-400/15 text-emerald-300"
                                : index === activeStep
                                  ? "bg-violet-400/15 text-violet-200"
                                  : "bg-white/[0.05] text-slate-600"
                            }`}
                          >
                            {index < activeStep ? (
                              <Check size={10} />
                            ) : index === activeStep ? (
                              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-300" />
                            ) : (
                              <span className="h-1 w-1 rounded-full bg-current" />
                            )}
                          </span>
                          <span className={index === activeStep ? "text-slate-200" : "text-slate-500"}>
                            {step}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5, transition: { duration: 0.2 } }}
              transition={{ duration: 0.2 }}
              role="alert"
              className="mt-5 flex items-start gap-3 rounded-xl border border-rose-400/20 bg-rose-400/[0.07] p-4"
            >
              <AlertCircle size={17} className="mt-0.5 shrink-0 text-rose-300" />
              <div className="flex-1">
                <p className="text-sm font-medium text-rose-100">We couldn’t complete your analysis</p>
                <p className="mt-1 text-xs leading-5 text-rose-200/70">{error}</p>
              </div>
              <button
                onClick={() => setError("")}
                className="rounded-lg p-1 text-rose-200/60 transition hover:bg-white/5 hover:text-rose-100"
                aria-label="Dismiss error"
              >
                <X size={15} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {result && <Results result={result} />}

        <footer className="mt-14 flex flex-col items-center justify-between gap-3 border-t border-white/[0.06] pt-5 text-[11px] text-slate-600 sm:flex-row">
          <span>© 2025 orbit. Built for your next chapter.</span>
          <span className="flex items-center gap-2">
            <span className="h-1 w-1 rounded-full bg-emerald-400/70" />
            Your resume is processed securely and never stored
          </span>
        </footer>
      </main>
    </div>
  );
}

interface ResultsProps {
  result: MatchResponse;
}

function Results({ result }: ResultsProps) {
  return (
    <motion.section
      id="results"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="mt-12 scroll-mt-8 space-y-8"
    >
      <header>
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-700">
          <BadgeCheck size={16} />
          Analysis complete
        </div>
        <h2 className="text-2xl font-semibold tracking-[-0.04em] text-slate-900 sm:text-[28px]">
          Your match report
        </h2>
        <p className="mt-1.5 text-sm text-slate-600">
          Your experience, matched to where you can make an impact.
        </p>
      </header>

      <section aria-labelledby="candidate-summary-heading">
        <SectionHeading id="candidate-summary-heading" number="A" title="Candidate summary" />
        <ProfileCard resume={result.resume} />
      </section>

      <section aria-labelledby="internship-matches-heading">
        <SectionHeading id="internship-matches-heading" number="B" title="Top internship matches" />
        <div className="panel p-5 sm:p-6">
          <div className="mb-5 flex items-center justify-between gap-3">
            <p className="text-sm text-slate-600">Ranked by your skills and experience</p>
            <span className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-[11px] text-slate-600">
              {result.top_internships.length} {result.top_internships.length === 1 ? "match" : "matches"}
            </span>
          </div>
          {result.top_internships.length ? (
            <div className="space-y-3">
              {result.top_internships.map((match, index) => (
                <InternshipCard key={`${match.internship.job_id}-${index}`} match={match} rank={index + 1} />
              ))}
            </div>
          ) : (
            <EmptyMatches />
          )}
        </div>
      </section>

      <section aria-labelledby="recruiter-roadmap-heading">
        <SectionHeading id="recruiter-roadmap-heading" number="C" title="Recruiter critique & learning roadmap" />
        <div className="panel p-5 sm:p-6">
          <RecruiterCard gapAnalysis={result.gap_analysis} />
          <div className="my-6 border-t border-slate-200" />
          <RoadmapCard gapAnalysis={result.gap_analysis} />
        </div>
      </section>
    </motion.section>
  );
}

function SectionHeading({ id, number, title }: { id: string; number: string; title: string }) {
  return (
    <h3 id={id} className="mb-3 flex items-center gap-3 text-lg font-semibold text-slate-900">
      <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-50 text-xs font-semibold text-indigo-700">
        {number}
      </span>
      {title}
    </h3>
  );
}

function ProfileCard({ resume }: { resume: Resume }) {
  return (
    <div className="panel p-5 sm:p-6">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <span className="card-icon card-icon-violet">
              <FileText size={15} />
            </span>
            Parsed candidate profile
          </div>
          <p className="ml-9 mt-1 text-xs text-slate-500">Extracted from your resume</p>
        </div>
        <span className="flex items-center gap-1.5 rounded-lg border border-emerald-300/10 bg-emerald-300/[0.045] px-2.5 py-1.5 text-[10px] font-medium text-emerald-200">
          <CheckCircle2 size={12} /> Parsed
        </span>
      </div>
      <div>
        <SectionLabel icon={<Code2 size={12} />} label={`Skills (${resume.skills.length})`} />
        {resume.skills.length ? (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {resume.skills.map((skill, index) => (
              <motion.span
                key={`${skill}-${index}`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: Math.min(index * 0.025, 0.2), duration: 0.2 }}
                className="skill-chip"
              >
                {skill}
              </motion.span>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">No skills were extracted.</p>
        )}
      </div>

      <div className="mt-5">
        <SectionLabel icon={<GraduationCap size={13} />} label="Education" />
        {resume.education.length ? (
          <div className="relative ml-[5px] mt-3 space-y-3 border-l border-white/[0.09] pl-4">
            {resume.education.map((education, index) => (
              <div key={`${education.institution}-${index}`} className="relative">
                <span className="absolute -left-[20px] top-1 h-2 w-2 rounded-full border-2 border-violet-300/70 bg-panel" />
                <p className="text-xs font-medium text-slate-200">{education.degree || "Education"}</p>
                <p className="mt-1 text-[11px] text-slate-500">
                  {education.institution}
                  {education.year ? ` · ${education.year}` : ""}
                  {education.cgpa ? ` · GPA ${education.cgpa}` : ""}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">No education details were extracted.</p>
        )}
      </div>

      <div className="mt-5">
        <SectionLabel icon={<Briefcase size={12} />} label="Experience" />
        {resume.experience.length ? (
          <div className="relative ml-[5px] mt-3 space-y-4 border-l border-slate-200 pl-4">
            {resume.experience.map((experience, index) => (
              <div key={`${experience.company}-${index}`} className="relative">
                <span className="absolute -left-[20px] top-1 h-2 w-2 rounded-full border-2 border-indigo-300 bg-white" />
                <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
                  <div>
                    <p className="text-xs font-semibold text-slate-800">{experience.role}</p>
                    <p className="mt-1 text-[11px] text-slate-600">{experience.company}</p>
                  </div>
                  <span className="text-[11px] text-slate-500">{experience.duration}</span>
                </div>
                {experience.description[0] && (
                  <p className="mt-2 text-xs leading-[1.65] text-slate-600">
                    {experience.description[0]}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">No work experience was extracted.</p>
        )}
      </div>

      {resume.projects.length > 0 && (
        <div className="mt-5 border-t border-white/[0.06] pt-4">
          <SectionLabel icon={<Code2 size={12} />} label={`Projects (${resume.projects.length})`} />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {resume.projects.map((project, index) => (
              <span key={`${project.name}-${index}`} className="project-chip">
                {project.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function InternshipCard({ match, rank }: { match: InternshipMatch; rank: number }) {
  const { internship, match_score: score } = match;
  const percent = Math.round(score.overall_score * 100);
  const missingSkills = [...score.missing_required, ...score.missing_preferred];

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(rank * 0.025, 0.1), duration: 0.2 }}
      className="match-card"
    >
      <div className="flex items-start gap-3">
        <span className="rank-number">{String(rank).padStart(2, "0")}</span>
        <span className="company-avatar">
          <Briefcase size={15} />
        </span>
        <div className="min-w-0 flex-1 pt-0.5">
          <h3 className="text-sm font-semibold text-slate-900">{internship.title || "Internship role"}</h3>
          <p className="mt-1 text-xs text-slate-600">{internship.company || "Company not specified"}</p>
        </div>
        <span className={`score-badge ${percent >= 75 ? "score-badge-good" : ""}`}>
          <span className="text-base font-semibold">{percent}%</span>
          <span className="text-[9px] uppercase tracking-wide">match</span>
        </span>
      </div>
      <div className="ml-12 mt-3 flex flex-wrap gap-x-3 gap-y-1.5">
          {internship.location && (
            <span className="inline-flex items-center gap-1 text-xs text-slate-600">
              <MapPin size={11} /> {internship.location}
            </span>
          )}
          {internship.work_mode && (
            <span className="rounded border border-slate-200 px-1.5 py-0.5 text-[10px] text-slate-600">
              {internship.work_mode}
            </span>
          )}
          {internship.duration && (
            <span className="inline-flex items-center gap-1 text-xs text-slate-600">
              <Clock3 size={11} /> {internship.duration}
            </span>
          )}
      </div>
      {internship.description && (
        <p className="ml-12 mt-3 text-xs leading-5 text-slate-600">{internship.description}</p>
      )}
      <div className="ml-12 mt-4 grid gap-3 border-t border-slate-200 pt-4 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">
            Matching skills
          </p>
          <div className="flex flex-wrap gap-1.5">
            {[...score.matched_required, ...score.matched_preferred].length ? (
              [...score.matched_required, ...score.matched_preferred].map((skill) => (
                <span key={skill} className="skill-chip skill-chip-match">
                  <Check size={10} /> {skill}
                </span>
              ))
            ) : (
              <span className="text-xs text-slate-500">No matching skills listed.</span>
            )}
          </div>
        </div>
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">
            Skills to develop
          </p>
          <div className="flex flex-wrap gap-1.5">
            {missingSkills.length ? (
              missingSkills.map((skill) => (
                <span key={skill} className="skill-chip skill-chip-warn">{skill}</span>
              ))
            ) : (
              <span className="text-xs text-emerald-700">No skill gaps identified.</span>
            )}
          </div>
        </div>
      </div>
      {internship.eligibility && (
        <p className="ml-12 mt-3 text-xs leading-5 text-slate-600">
          <span className="font-medium text-slate-700">Eligibility: </span>{internship.eligibility}
        </p>
      )}
    </motion.article>
  );
}

function RecruiterCard({ gapAnalysis }: { gapAnalysis: GapAnalysis | null }) {
  return (
    <div className="panel relative overflow-hidden p-5 sm:p-6">
      <div className="relative">
        <div className="flex items-center gap-2 text-sm font-medium text-white">
          <span className="card-icon card-icon-amber">
            <Sparkles size={15} />
          </span>
          AI recruiter’s take
        </div>
        <p className="ml-9 mt-1 text-xs text-slate-500">A thoughtful perspective on your next step</p>
        {gapAnalysis ? (
          <>
            <div className="mt-5 rounded-xl border border-white/[0.06] bg-black/10 p-4">
              <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-violet-200/80">
                <Lightbulb size={12} /> Recruiter critique
              </div>
              <p className="text-xs leading-[1.8] text-slate-300">{gapAnalysis.resume_critique}</p>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                Worth developing
              </span>
              {gapAnalysis.missing_skills.length ? (
                gapAnalysis.missing_skills.slice(0, 5).map((skill) => (
                  <span key={skill} className="skill-chip skill-chip-warn">{skill}</span>
                ))
              ) : (
                <span className="text-xs text-emerald-300">No gaps identified</span>
              )}
            </div>
          </>
        ) : (
          <div className="mt-5 rounded-xl border border-white/[0.06] bg-black/10 p-4">
            <p className="text-xs leading-6 text-slate-400">
              Recruiter insights are unavailable because no internship matched this profile. Seed the
              internship database to unlock personalized feedback and a learning roadmap.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function RoadmapCard({ gapAnalysis }: { gapAnalysis: GapAnalysis | null }) {
  return (
    <div className="panel p-5 sm:p-6">
      <div className="mb-6 flex items-center gap-3">
        <span className="card-icon card-icon-violet">
          <Rocket size={15} />
        </span>
        <div>
          <h3 className="text-sm font-medium text-white">Your learning roadmap</h3>
          <p className="mt-1 text-xs text-slate-500">Small steps, meaningful momentum.</p>
        </div>
      </div>
      {gapAnalysis?.learning_path.length ? (
        <div className="relative ml-3 border-l border-violet-300/15 pl-6">
          {gapAnalysis.learning_path.map((item, index) => (
            <motion.div
              key={`${item.topic}-${index}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: Math.min(index * 0.025, 0.2), duration: 0.2 }}
              className="relative pb-6 last:pb-0"
            >
              <span className="absolute -left-[31px] top-1 flex h-[12px] w-[12px] items-center justify-center rounded-full border border-violet-300/45 bg-panel">
                <span className="h-1 w-1 rounded-full bg-violet-300" />
              </span>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h4 className="text-sm font-medium text-slate-200">{item.topic}</h4>
                  <span className={`resource-tag resource-${item.resource_type.toLowerCase()}`}>
                    {resourceIcon(item.resource_type)}
                    {item.resource_type || "Resource"}
                  </span>
                </div>
                <span className="flex items-center gap-1.5 text-[11px] text-slate-500">
                  <Clock3 size={12} /> {item.estimated_hours} hrs
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="empty-roadmap">
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl border border-emerald-300/10 bg-emerald-300/[0.06]">
            <BadgeCheck size={20} className="text-emerald-300" />
          </span>
          <p className="mt-4 text-sm font-medium text-slate-200">You’re in a great place</p>
          <p className="mt-1 max-w-xs text-center text-xs leading-5 text-slate-500">
            {gapAnalysis
              ? "No learning steps were recommended for this match."
              : "A learning roadmap will appear once internship matches are available."}
          </p>
        </div>
      )}
    </div>
  );
}

function EmptyMatches() {
  return (
    <div className="empty-matches">
      <div className="empty-orbit">
        <div className="empty-orbit-core"><Briefcase size={19} /></div>
        <span className="empty-orbit-dot" />
        <span className="empty-orbit-dot empty-orbit-dot-two" />
      </div>
      <p className="mt-5 text-sm font-medium text-slate-200">Your next match is out there</p>
      <p className="mt-1 max-w-[270px] text-center text-xs leading-5 text-slate-500">
        We couldn’t find a close fit just yet. Add opportunities to your database and check back.
      </p>
      <div className="mt-5 flex items-center gap-2 rounded-xl border border-amber-300/10 bg-amber-300/[0.045] px-3 py-2.5">
        <Plus size={14} className="text-amber-300" />
        <span className="text-[11px] text-amber-100/80">Seed your internship database to get started</span>
      </div>
    </div>
  );
}

function SectionLabel({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">
      <span className="text-slate-500">{icon}</span>
      {label}
    </div>
  );
}

function resourceIcon(resource: string) {
  switch (resource.toLowerCase()) {
    case "course":
      return <GraduationCap size={11} />;
    case "docs":
    case "documentation":
      return <BookOpen size={11} />;
    case "project":
      return <Code2 size={11} />;
    default:
      return <ArrowDown size={11} />;
  }
}

export default App;
