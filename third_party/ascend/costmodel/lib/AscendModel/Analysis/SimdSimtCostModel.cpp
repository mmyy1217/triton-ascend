//===- SimdSimtCostModel.cpp - Ascend SIMD/SIMT candidate model ----------===//
//
// The numerical model in this file is the versioned C++ candidate model.  It
// intentionally produces a relative per-program selection score, not an
// end-to-end kernel-time prediction.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/SimdSimtCostModel.h"
#include "AscendModel/Analysis/MicrobenchmarkProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

constexpr llvm::StringLiteral kAllSimd = "all_simd";
constexpr llvm::StringLiteral kAllSimtOnly = "all_simt_only";
constexpr llvm::StringLiteral kMixedSimdSimt = "mixed_simd_simt";

struct OpProfile {
  double throughput = 0.0;
  double factor = 1.0;
  std::string confidence = "none";
};

struct CoverageProfile {
  double minimumIrregularDensity = 0.0;
  int64_t tinyDotFlopsMax = 0;
  int64_t tinyDotMaxTensorNumel = 0;
  int64_t rowwiseLoopTripSumMax = 0;
  int64_t rowwiseMaskRankSumMax = 0;
  int64_t rowwiseWeightedReductionsMax = 0;
  int64_t rowwiseMaxTensorNumel = 0;
  int64_t rank1WeightedReductionsMax = 0;
  int64_t rank1MaxTensorNumel = 0;
};

struct StructuralProfile {
  double irregularPerDensity = 0.0;
  double irregularCap = 0.0;
  double tinyDotIrregularPerDensity = 0.0;
  double tinyDotIrregularCap = 0.0;
  double perMaskRank = 0.0;
  double maskCap = 0.0;
  double perWeightedReduction = 0.0;
  double reductionCap = 0.0;
  double perStaticLoopTrip = 0.0;
  double loopCap = 0.0;
  double controlFlow = 0.0;
  double rank1IndirectVectorReduction = 0.0;
  double tinyDot = 0.0;
  int64_t tinyDotFlopsMax = 0;
};

struct MixedBlendProfile {
  double base = 0.0;
  double perStaticLoopTrip = 0.0;
  double loopCap = 0.0;
  double perMaskBroadcast = 0.0;
  double maskCap = 0.0;
  double perWeightedReduction = 0.0;
  double reductionCap = 0.0;
  double controlFlow = 0.0;
  double tinyDot = 0.0;
  double max = 0.0;
};

struct TransitionProfile {
  int64_t numWarps = 0;
  double emptySimtSetupCycles = 0.0;
};

struct CandidateProfile {
  std::string profileVersion;
  std::string target;
  std::vector<std::string> compatibleTargets;
  std::string scoreUnit;
  std::string minimumConfidence = "medium";
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;

  double programIssueScale = 1.0;
  std::string rankingConfidence = "low";
  double tinyDotMixedPenaltyAtZero = 0.0;
  std::string calibrationSource;
  CoverageProfile coverage;
  StructuralProfile structural;
  MixedBlendProfile mixedBlend;

  int64_t simdVectorWidthBits = 2048;
  double simdSetupCycles = 0.0;
  llvm::StringMap<OpProfile> simdOps;
  double simdMte2BytesPerCycle = 0.0;
  double simdMte3BytesPerCycle = 0.0;
  std::string simdMemoryConfidence = "none";
  double simdDotSetupCycles = 0.0;
  double simdDotFlopsPerCycle = 0.0;
  std::string simdDotConfidence = "none";

  int64_t simtWarpSize = 32;
  double simtSetupCycles = 0.0;
  llvm::StringMap<OpProfile> simtOps;
  double simtDotSetupCycles = 0.0;
  double simtDotFlopsPerCycle = 0.0;
  std::string simtDotConfidence = "none";
  double simtPredicateRate = 0.0;
  double simtShuffleRate = 0.0;
  std::string simtShuffleConfidence = "none";
  double simtLoadWarpRate = 0.0;
  double simtStoreWarpRate = 0.0;
  std::string simtMemoryConfidence = "none";
  std::vector<TransitionProfile> transitions;
  std::string transitionConfidence = "none";
};

/// Small fail-fast facade around llvm::json.  It permits a readable profile
/// parser while retaining a single actionable error message.
class ProfileJSONReader {
public:
  const llvm::json::Object *object(const llvm::json::Object &parent,
                                   llvm::StringRef key,
                                   llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getObject(key))
      return value;
    setError(context + "." + key + " must be an object");
    return nullptr;
  }

  const llvm::json::Array *array(const llvm::json::Object &parent,
                                 llvm::StringRef key,
                                 llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getArray(key))
      return value;
    setError(context + "." + key + " must be an array");
    return nullptr;
  }

  double number(const llvm::json::Object &parent, llvm::StringRef key,
                llvm::StringRef context) {
    if (failed())
      return 0.0;
    if (auto value = parent.getNumber(key))
      return *value;
    setError(context + "." + key + " must be a number");
    return 0.0;
  }

  int64_t integer(const llvm::json::Object &parent, llvm::StringRef key,
                  llvm::StringRef context) {
    if (failed())
      return 0;
    if (auto value = parent.getInteger(key))
      return *value;
    setError(context + "." + key + " must be an integer");
    return 0;
  }

  std::string string(const llvm::json::Object &parent, llvm::StringRef key,
                     llvm::StringRef context) {
    if (failed())
      return {};
    if (auto value = parent.getString(key))
      return value->str();
    setError(context + "." + key + " must be a string");
    return {};
  }

  std::string optionalString(const llvm::json::Object &parent,
                             llvm::StringRef key,
                             llvm::StringRef defaultValue = {}) {
    if (auto value = parent.getString(key))
      return value->str();
    return defaultValue.str();
  }

  double optionalNumber(const llvm::json::Object &parent, llvm::StringRef key,
                        double defaultValue) {
    if (auto value = parent.getNumber(key))
      return *value;
    return defaultValue;
  }

  bool failed() const { return !error.empty(); }
  llvm::StringRef getError() const { return error; }

  void setError(const llvm::Twine &message) {
    if (error.empty())
      error = message.str();
  }

private:
  std::string error;
};

static llvm::json::Object toJSON(const llvm::StringMap<int64_t> &values) {
  llvm::json::Object result;
  for (const auto &entry : values)
    result[entry.first()] = entry.second;
  return result;
}

static llvm::json::Object toJSON(const llvm::StringMap<double> &values) {
  llvm::json::Object result;
  for (const auto &entry : values)
    result[entry.first()] = entry.second;
  return result;
}

static void initializeWorkMaps(SimdSimtFeatureSummary &features) {
  for (llvm::StringRef key :
       {"load", "store", "reduce", "scan", "gather", "histogram", "atomic",
        "add", "sub", "mul", "div", "max", "abs", "cmp", "select", "cast",
        "clamp"}) {
    features.weightedOps[key] = 0;
    features.opElements[key] = 0;
  }
}

static std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  os.flush();
  return text;
}

static bool isPointerType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

static Type getElementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

static int64_t parseTypeBitWidth(llvm::StringRef text) {
  for (size_t index = 0; index + 1 < text.size(); ++index) {
    if (text[index] != 'f' && text[index] != 'i')
      continue;
    size_t end = index + 1;
    while (end < text.size() && llvm::isDigit(text[end]))
      ++end;
    if (end == index + 1)
      continue;
    int64_t width = 0;
    if (!text.slice(index + 1, end).getAsInteger(10, width) && width > 0)
      return width;
  }
  return 0;
}

static int64_t getTypeBitWidth(Type type, int64_t defaultWidth = 32) {
  type = getElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  int64_t parsed = parseTypeBitWidth(typeToString(type));
  return parsed > 0 ? parsed : defaultWidth;
}

static bool isMaskTensorType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return false;
  auto integer = dyn_cast<IntegerType>(tensor.getElementType());
  return integer && integer.getWidth() == 1;
}

static int64_t getStaticNumElements(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return 1;
  int64_t count = 1;
  for (int64_t dim : tensor.getShape()) {
    if (ShapedType::isDynamic(dim) || dim <= 0)
      return 1;
    if (count > std::numeric_limits<int64_t>::max() / dim)
      return std::numeric_limits<int64_t>::max();
    count *= dim;
  }
  return std::max<int64_t>(1, count);
}

static int64_t getOperationElements(Operation *op) {
  int64_t elements = 1;
  for (Type type : op->getOperandTypes())
    elements = std::max(elements, getStaticNumElements(type));
  for (Type type : op->getResultTypes())
    elements = std::max(elements, getStaticNumElements(type));
  return elements;
}

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  if (auto integer = definingOp->getAttrOfType<IntegerAttr>("value"))
    return integer.getInt();
  return std::nullopt;
}

static int64_t getStaticLoopTripCount(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumOperands() < 3)
    return 1;
  auto lower = getConstantInteger(op->getOperand(0));
  auto upper = getConstantInteger(op->getOperand(1));
  auto step = getConstantInteger(op->getOperand(2));
  if (!lower || !upper || !step || *step == 0)
    return 1;
  int64_t span = *upper - *lower;
  if (span > 0 && *step > 0)
    return std::max<int64_t>(1, (span + *step - 1) / *step);
  if (span < 0 && *step < 0) {
    int64_t positiveSpan = -span;
    int64_t positiveStep = -*step;
    return std::max<int64_t>(
        1, (positiveSpan + positiveStep - 1) / positiveStep);
  }
  return 1;
}

static int64_t getLoopMultiplier(Operation *op) {
  int64_t multiplier = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "scf.for")
      continue;
    int64_t tripCount = getStaticLoopTripCount(parent);
    if (tripCount > 0 &&
        multiplier <= std::numeric_limits<int64_t>::max() / tripCount)
      multiplier *= tripCount;
  }
  return multiplier;
}

static bool isCastOp(llvm::StringRef name) {
  return name.starts_with("arith.ext") || name.starts_with("arith.trunc") ||
         name == "arith.sitofp" || name == "arith.uitofp" ||
         name == "arith.fptosi" || name == "arith.fptoui" ||
         name.starts_with("arith.index_cast");
}

static llvm::StringRef classifyWeightedOp(llvm::StringRef name) {
  if (name == "tt.load")
    return "load";
  if (name == "tt.store")
    return "store";
  if (name == "tt.reduce")
    return "reduce";
  if (name == "tt.scan" || name == "tt.associative_scan")
    return "scan";
  if (name == "tt.gather")
    return "gather";
  if (name == "tt.histogram")
    return "histogram";
  if (name.starts_with("tt.atomic"))
    return "atomic";
  if (name == "arith.addf" || name == "arith.addi")
    return "add";
  if (name == "arith.subf" || name == "arith.subi")
    return "sub";
  if (name == "arith.mulf" || name == "arith.muli")
    return "mul";
  if (name == "arith.divf" || name == "arith.divsi" ||
      name == "arith.divui")
    return "div";
  if (name == "arith.maxnumf" || name == "arith.maxf" ||
      name == "arith.maxsi" || name == "arith.maxui")
    return "max";
  if (name == "math.absf" || name == "math.absi")
    return "abs";
  if (name == "arith.cmpf" || name == "arith.cmpi")
    return "cmp";
  if (name == "arith.select")
    return "select";
  if (isCastOp(name))
    return "cast";
  if (name.starts_with("tt.clamp"))
    return "clamp";
  return {};
}

static bool isPlainOneDimensionalCumsum(Operation *scan) {
  if (!scan || scan->getName().getStringRef() != "tt.scan" ||
      scan->getNumOperands() == 0)
    return false;
  auto input = dyn_cast<RankedTensorType>(scan->getOperand(0).getType());
  auto axis = scan->getAttrOfType<IntegerAttr>("axis");
  if (!input || !axis)
    return false;
  int64_t axisValue = axis.getInt();
  if (axisValue < 0 || axisValue >= input.getRank())
    return false;
  for (auto [index, extent] : llvm::enumerate(input.getShape()))
    if (static_cast<int64_t>(index) != axisValue && extent != 1)
      return false;

  int64_t realOps = 0;
  bool isAdd = false;
  scan->walk([&](Operation *nested) {
    if (nested == scan)
      return;
    llvm::StringRef name = nested->getName().getStringRef();
    if (name == "tt.scan.return" || name == "arith.extf" ||
        name == "arith.truncf" || name == "arith.bitcast")
      return;
    ++realOps;
    isAdd = name == "arith.addf" || name == "arith.addi";
  });
  return realOps == 1 && isAdd;
}

static void appendUnique(std::vector<std::string> &values,
                         llvm::StringRef value) {
  if (llvm::find(values, value.str()) == values.end())
    values.push_back(value.str());
}

static int confidenceRank(llvm::StringRef confidence) {
  if (confidence == "high")
    return 3;
  if (confidence == "medium")
    return 2;
  if (confidence == "low")
    return 1;
  return 0;
}

static std::string minimumConfidence(llvm::ArrayRef<std::string> values) {
  if (values.empty())
    return "none";
  return *std::min_element(
      values.begin(), values.end(),
      [](const std::string &lhs, const std::string &rhs) {
        return confidenceRank(lhs) < confidenceRank(rhs);
      });
}

static bool wildcardMatch(llvm::StringRef pattern, llvm::StringRef value) {
  size_t patternIndex = 0;
  size_t valueIndex = 0;
  size_t starIndex = llvm::StringRef::npos;
  size_t retryValueIndex = 0;
  while (valueIndex < value.size()) {
    if (patternIndex < pattern.size() &&
        (pattern[patternIndex] == '?' ||
         pattern[patternIndex] == value[valueIndex])) {
      ++patternIndex;
      ++valueIndex;
      continue;
    }
    if (patternIndex < pattern.size() && pattern[patternIndex] == '*') {
      starIndex = patternIndex++;
      retryValueIndex = valueIndex;
      continue;
    }
    if (starIndex != llvm::StringRef::npos) {
      patternIndex = starIndex + 1;
      valueIndex = ++retryValueIndex;
      continue;
    }
    return false;
  }
  while (patternIndex < pattern.size() && pattern[patternIndex] == '*')
    ++patternIndex;
  return patternIndex == pattern.size();
}

static bool targetMatches(const CandidateProfile &profile,
                          llvm::StringRef actualTarget) {
  if (actualTarget.trim().empty())
    return true;
  std::string actual = actualTarget.trim().lower();
  std::vector<std::string> patterns = profile.compatibleTargets;
  patterns.push_back(profile.target);
  for (std::string pattern : patterns) {
    std::replace(pattern.begin(), pattern.end(), ':', '/');
    llvm::SmallVector<llvm::StringRef> aliases;
    llvm::StringRef(pattern).split(aliases, '/', -1, false);
    for (llvm::StringRef alias : aliases) {
      std::string lower = alias.trim().lower();
      if (!lower.empty() && wildcardMatch(lower, actual))
        return true;
    }
  }
  return false;
}

static double resolveNumberOrMeasurement(
    const llvm::json::Object &object, llvm::StringRef numberKey,
    llvm::StringRef measurementKey, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context, std::string *measurementConfidence = nullptr) {
  if (auto reference = object.getString(measurementKey)) {
    if (!microbench) {
      reader.setError(context + "." + measurementKey +
                      " requires microbenchmark_profile");
      return 0.0;
    }
    llvm::StringRef expectedCycleDomain = "none";
    if (expectedUnit == "system_cycle" ||
        expectedUnit.ends_with("/system_cycle"))
      expectedCycleDomain = "SYS_CNT";
    auto value = microbench->requireValue(*reference, expectedUnit,
                                          expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return 0.0;
    }
    if (measurementConfidence) {
      const MicrobenchmarkMeasurement *measurement =
          microbench->getMeasurement(*reference);
      *measurementConfidence =
          measurement ? measurement->confidence : "none";
    }
    return *value;
  }
  return reader.number(object, numberKey, context);
}

static OpProfile resolveOpProfile(const llvm::json::Object &ops,
                                  llvm::StringRef opName,
                                  llvm::StringRef throughputKey,
                                  llvm::StringRef expectedUnit,
                                  const MicrobenchmarkProfile *microbench,
                                  ProfileJSONReader &reader) {
  OpProfile result;
  const llvm::json::Value *raw = ops.get(opName);
  if (!raw) {
    reader.setError("missing operation profile " + opName);
    return result;
  }
  const auto *op = raw->getAsObject();
  if (!op) {
    reader.setError("operation profile " + opName + " must be an object");
    return result;
  }
  if (auto relative = op->getString("relative_to")) {
    OpProfile base = resolveOpProfile(ops, *relative, throughputKey,
                                      expectedUnit, microbench, reader);
    result.throughput = base.throughput;
    result.factor = reader.optionalNumber(*op, "factor", 1.0);
    result.confidence = reader.optionalString(*op, "confidence", "low");
    return result;
  }
  std::string measuredConfidence = "none";
  result.throughput = resolveNumberOrMeasurement(
      *op, throughputKey, "throughput_measurement", expectedUnit, microbench,
      reader, opName, &measuredConfidence);
  result.factor = reader.optionalNumber(*op, "factor", 1.0);
  result.confidence =
      reader.optionalString(*op, "confidence", measuredConfidence);
  return result;
}

/// Match Python's json.dumps(value, sort_keys=True) representation.  Keeping
/// this stable makes profile_content_sha256 identical across the temporary
/// Python model and this C++ implementation.
static void emitPythonCanonicalJSON(const llvm::json::Value &value,
                                    llvm::raw_ostream &os) {
  if (const auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);
    os << '{';
    bool first = true;
    for (llvm::StringRef key : keys) {
      if (!first)
        os << ", ";
      first = false;
      os << llvm::json::Value(key.str()) << ": ";
      emitPythonCanonicalJSON(*object->get(key), os);
    }
    os << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    os << '[';
    bool first = true;
    for (const llvm::json::Value &element : *array) {
      if (!first)
        os << ", ";
      first = false;
      emitPythonCanonicalJSON(element, os);
    }
    os << ']';
    return;
  }
  os << value;
}

static std::string resolveProfileReference(llvm::StringRef ownerPath,
                                           llvm::StringRef reference) {
  if (llvm::sys::path::is_absolute(reference))
    return reference.str();
  llvm::SmallString<256> resolved(ownerPath);
  llvm::sys::path::remove_filename(resolved);
  llvm::sys::path::append(resolved, reference);
  llvm::sys::path::remove_dots(resolved, true);
  return resolved.str().str();
}

static llvm::Expected<CandidateProfile>
loadCandidateProfile(llvm::StringRef requestedPath) {
  std::string path =
      requestedPath.empty() ? getDefaultSimdSimtProfilePath()
                            : requestedPath.str();
  if (path.empty())
    return llvm::createStringError(
        std::errc::no_such_file_or_directory,
        "SIMD/SIMT profile path is empty; set "
        "TRITON_ASCEND_SIMD_SIMT_PROFILE");

  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(
        buffer.getError(), "failed to read SIMD/SIMT profile '%s'",
        path.c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to parse SIMD/SIMT profile '%s': %s", path.c_str(),
        llvm::toString(parsed.takeError()).c_str());
  const auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile root must be an object");
  auto selectionSchemaVersion = root->getInteger("schema_version");
  if (!selectionSchemaVersion ||
      (*selectionSchemaVersion != 1 && *selectionSchemaVersion != 2))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile schema_version must be 1 or 2");

  CandidateProfile profile;
  ProfileJSONReader reader;
  std::optional<MicrobenchmarkProfile> microbenchmarkProfile;
  if (auto reference = root->getString("microbenchmark_profile")) {
    std::string resolved = resolveProfileReference(path, *reference);
    auto loaded = MicrobenchmarkProfile::loadFromFile(resolved);
    if (!loaded)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to load shared microbenchmark profile referenced by '%s': %s",
          path.c_str(), llvm::toString(loaded.takeError()).c_str());
    microbenchmarkProfile.emplace(std::move(*loaded));
    profile.microbenchmarkProfileVersion =
        microbenchmarkProfile->getProfileVersion().str();
    profile.microbenchmarkProfileTarget =
        microbenchmarkProfile->getTarget().str();
    profile.microbenchmarkContentSha256 =
        microbenchmarkProfile->getContentSha256().str();
  }
  const MicrobenchmarkProfile *microbench =
      microbenchmarkProfile ? &*microbenchmarkProfile : nullptr;

  profile.profileVersion =
      reader.string(*root, "profile_version", "profile");
  profile.target = reader.string(*root, "target", "profile");
  if (microbench &&
      llvm::StringRef(profile.target) != microbench->getTarget())
    reader.setError("selection profile target '" + profile.target +
                    "' does not match shared microbenchmark target '" +
                    microbench->getTarget().str() + "'");
  profile.scoreUnit = reader.string(*root, "score_unit", "profile");
  if (const auto *targets =
          reader.array(*root, "compatible_targets", "profile")) {
    for (const llvm::json::Value &target : *targets) {
      if (auto text = target.getAsString())
        profile.compatibleTargets.push_back(text->str());
      else
        reader.setError("profile.compatible_targets entries must be strings");
    }
  }
  if (const auto *policy = reader.object(*root, "policy", "profile"))
    profile.minimumConfidence =
        reader.optionalString(*policy, "minimum_confidence_for_decision",
                              "medium");

  const auto *calibration =
      reader.object(*root, "selection_calibration", "profile");
  if (calibration) {
    profile.programIssueScale =
        reader.number(*calibration, "program_issue_scale",
                      "profile.selection_calibration");
    profile.rankingConfidence =
        reader.optionalString(*calibration, "ranking_confidence", "low");
    profile.tinyDotMixedPenaltyAtZero =
        reader.number(*calibration, "tiny_dot_mixed_penalty_ratio_at_zero",
                      "profile.selection_calibration");
    profile.calibrationSource =
        reader.optionalString(*calibration, "source", "");

    if (const auto *coverage =
            reader.object(*calibration, "coverage",
                          "profile.selection_calibration")) {
      profile.coverage.minimumIrregularDensity =
          reader.number(*coverage, "minimum_irregular_density", "coverage");
      profile.coverage.tinyDotFlopsMax =
          reader.integer(*coverage, "tiny_dot_flops_max", "coverage");
      profile.coverage.tinyDotMaxTensorNumel =
          reader.integer(*coverage, "tiny_dot_max_tensor_numel", "coverage");
      profile.coverage.rowwiseLoopTripSumMax =
          reader.integer(*coverage, "rowwise_loop_trip_sum_max", "coverage");
      profile.coverage.rowwiseMaskRankSumMax =
          reader.integer(*coverage, "rowwise_mask_rank_sum_max", "coverage");
      profile.coverage.rowwiseWeightedReductionsMax = reader.integer(
          *coverage, "rowwise_weighted_reductions_max", "coverage");
      profile.coverage.rowwiseMaxTensorNumel =
          reader.integer(*coverage, "rowwise_max_tensor_numel", "coverage");
      profile.coverage.rank1WeightedReductionsMax = reader.integer(
          *coverage, "rank1_weighted_reductions_max", "coverage");
      profile.coverage.rank1MaxTensorNumel =
          reader.integer(*coverage, "rank1_max_tensor_numel", "coverage");
    }

    if (const auto *structural =
            reader.object(*calibration, "simd_structural_penalty_ratio",
                          "profile.selection_calibration")) {
      profile.structural.irregularPerDensity =
          reader.number(*structural, "irregular_per_density", "structural");
      profile.structural.irregularCap =
          reader.number(*structural, "irregular_cap", "structural");
      profile.structural.tinyDotIrregularPerDensity = reader.number(
          *structural, "tiny_dot_irregular_per_density", "structural");
      profile.structural.tinyDotIrregularCap =
          reader.number(*structural, "tiny_dot_irregular_cap", "structural");
      profile.structural.perMaskRank =
          reader.number(*structural, "per_mask_rank", "structural");
      profile.structural.maskCap =
          reader.number(*structural, "mask_cap", "structural");
      profile.structural.perWeightedReduction = reader.number(
          *structural, "per_weighted_reduction", "structural");
      profile.structural.reductionCap =
          reader.number(*structural, "reduction_cap", "structural");
      profile.structural.perStaticLoopTrip = reader.number(
          *structural, "per_static_loop_trip", "structural");
      profile.structural.loopCap =
          reader.number(*structural, "loop_cap", "structural");
      profile.structural.controlFlow =
          reader.number(*structural, "control_flow", "structural");
      profile.structural.rank1IndirectVectorReduction =
          reader.number(*structural, "rank1_indirect_vector_reduction",
                        "structural");
      profile.structural.tinyDot =
          reader.number(*structural, "tiny_dot", "structural");
      profile.structural.tinyDotFlopsMax =
          reader.integer(*structural, "tiny_dot_flops_max", "structural");
    }

    if (const auto *mixed =
            reader.object(*calibration, "mixed_simd_fraction",
                          "profile.selection_calibration")) {
      profile.mixedBlend.base = reader.number(*mixed, "base", "mixed");
      profile.mixedBlend.perStaticLoopTrip =
          reader.number(*mixed, "per_static_loop_trip", "mixed");
      profile.mixedBlend.loopCap =
          reader.number(*mixed, "loop_cap", "mixed");
      profile.mixedBlend.perMaskBroadcast =
          reader.number(*mixed, "per_mask_broadcast", "mixed");
      profile.mixedBlend.maskCap =
          reader.number(*mixed, "mask_cap", "mixed");
      profile.mixedBlend.perWeightedReduction =
          reader.number(*mixed, "per_weighted_reduction", "mixed");
      profile.mixedBlend.reductionCap =
          reader.number(*mixed, "reduction_cap", "mixed");
      profile.mixedBlend.controlFlow =
          reader.number(*mixed, "control_flow", "mixed");
      profile.mixedBlend.tinyDot =
          reader.number(*mixed, "tiny_dot", "mixed");
      profile.mixedBlend.max = reader.number(*mixed, "max", "mixed");
    }
  }

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    if (simd->getString("vector_width_measurement")) {
      profile.simdVectorWidthBits = static_cast<int64_t>(std::llround(
          resolveNumberOrMeasurement(
              *simd, "vector_width_bits", "vector_width_measurement", "bit",
              microbench, reader, "simd")));
    } else {
      profile.simdVectorWidthBits =
          reader.integer(*simd, "vector_width_bits", "simd");
    }
    if (const auto *startup =
            reader.object(*simd, "startup_system_cycles", "simd"))
      profile.simdSetupCycles =
          reader.number(*startup, "vector", "simd.startup_system_cycles");
    if (const auto *ops = reader.object(*simd, "ops", "simd")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max",
            "f32.abs", "f32.exp", "f32.log", "predicate.cmp",
            "predicate.select", "convert.cast", "f32.clamp"})
        profile.simdOps[op] = resolveOpProfile(
            *ops, op, "throughput_vector_instructions_per_system_cycle",
            "vector_instruction/system_cycle", microbench, reader);
    }
    if (const auto *memory = reader.object(*simd, "memory", "simd")) {
      profile.simdMte2BytesPerCycle = reader.number(
          *memory, "vector_mte2_bytes_per_system_cycle", "simd.memory");
      profile.simdMte3BytesPerCycle =
          reader.number(*memory, "mte3_bytes_per_system_cycle", "simd.memory");
      profile.simdMemoryConfidence =
          reader.optionalString(*memory, "confidence", "none");
    }
    if (const auto *dot = reader.object(*simd, "dot", "simd")) {
      profile.simdDotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simd.dot");
      profile.simdDotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simd.dot");
      profile.simdDotConfidence =
          reader.optionalString(*dot, "confidence", "none");
    }
  }

  const auto *simt = reader.object(*root, "simt", "profile");
  if (simt) {
    if (simt->getString("warp_size_measurement")) {
      profile.simtWarpSize = static_cast<int64_t>(std::llround(
          resolveNumberOrMeasurement(
              *simt, "warp_size", "warp_size_measurement", "lane", microbench,
              reader, "simt")));
    } else {
      profile.simtWarpSize = reader.integer(*simt, "warp_size", "simt");
    }
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      std::string setupConfidence;
      profile.simtSetupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles", &setupConfidence);
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max",
            "f32.abs", "f32.exp", "f32.log", "predicate.cmp",
            "predicate.select", "convert.cast", "f32.clamp"})
        profile.simtOps[op] = resolveOpProfile(
            *ops, op, "throughput_scalar_ops_per_system_cycle",
            "scalar_op/system_cycle", microbench, reader);
    }
    if (const auto *dot = reader.object(*simt, "dot", "simt")) {
      profile.simtDotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simt.dot");
      profile.simtDotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simt.dot");
      profile.simtDotConfidence =
          reader.optionalString(*dot, "confidence", "none");
    }
    if (const auto *camodel =
            reader.object(*simt, "camodel_effective", "simt")) {
      if (const auto *rates = reader.object(
              *camodel, "warp_instructions_per_system_cycle",
              "simt.camodel_effective"))
        profile.simtPredicateRate =
            reader.number(*rates, "predicate", "simt.camodel_effective.rates");
    }
    if (const auto *shuffle = reader.object(*simt, "shuffle", "simt")) {
      std::string measuredConfidence;
      profile.simtShuffleRate = resolveNumberOrMeasurement(
          *shuffle, "warp_instructions_per_system_cycle",
          "throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.shuffle", &measuredConfidence);
      profile.simtShuffleConfidence = reader.optionalString(
          *shuffle, "confidence", measuredConfidence);
    }
    if (const auto *memory = reader.object(*simt, "memory", "simt")) {
      std::string loadConfidence;
      std::string storeConfidence;
      profile.simtLoadWarpRate = resolveNumberOrMeasurement(
          *memory, "load_warp_instructions_per_system_cycle",
          "load_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory", &loadConfidence);
      profile.simtStoreWarpRate = resolveNumberOrMeasurement(
          *memory, "store_warp_instructions_per_system_cycle",
          "store_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory", &storeConfidence);
      profile.simtMemoryConfidence = reader.optionalString(
          *memory, "confidence",
          minimumConfidence({loadConfidence, storeConfidence}));
    }
    if (const auto *transition =
            reader.object(*simt, "transition", "simt")) {
      for (int64_t numWarps : {1, 2, 4, 8, 16, 32}) {
        std::string key = std::to_string(numWarps);
        const auto *entry = transition->getObject(key);
        if (!entry)
          continue;
        std::string measuredConfidence;
        profile.transitions.push_back({
            numWarps,
            resolveNumberOrMeasurement(
                *entry, "empty_simt_setup_system_cycles", "measurement",
                "system_cycle", microbench, reader,
                "simt.transition." + key, &measuredConfidence)});
        if (profile.transitionConfidence == "none")
          profile.transitionConfidence = measuredConfidence;
      }
      profile.transitionConfidence = reader.optionalString(
          *transition, "confidence", profile.transitionConfidence);
    }
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (profile.profileVersion != "david-v100-simd-simt-20260727-v3" &&
      profile.profileVersion != "david-v100-simd-simt-20260727-v4" &&
      profile.profileVersion != "david-v100-simd-simt-20260728-v5")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "unsupported SIMD/SIMT profile version '%s' (expected v3, v4, or v5)",
        profile.profileVersion.c_str());
  if (profile.profileVersion == "david-v100-simd-simt-20260728-v5" &&
      !microbench)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT v5 profile must reference microbenchmark_profile");
  if (profile.profileVersion == "david-v100-simd-simt-20260728-v5" &&
      *selectionSchemaVersion != 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT v5 profile requires schema_version 2");
  if (profile.simdVectorWidthBits <= 0 || profile.simtWarpSize <= 0 ||
      profile.simdMte2BytesPerCycle <= 0.0 ||
      profile.simdMte3BytesPerCycle <= 0.0 ||
      profile.simtLoadWarpRate <= 0.0 ||
      profile.simtStoreWarpRate <= 0.0 ||
      profile.simtShuffleRate <= 0.0 || profile.simtPredicateRate <= 0.0 ||
      profile.transitions.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains non-positive rates or no transitions");

  std::string canonicalProfile;
  llvm::raw_string_ostream canonicalStream(canonicalProfile);
  emitPythonCanonicalJSON(*parsed, canonicalStream);
  canonicalStream.flush();
  llvm::ArrayRef<uint8_t> byteArray(
      reinterpret_cast<const uint8_t *>(canonicalProfile.data()),
      canonicalProfile.size());
  auto hash = llvm::SHA256::hash(byteArray);
  profile.selectionContentSha256 =
      llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);

  profile.contentSha256 = profile.selectionContentSha256;
  if (!profile.microbenchmarkContentSha256.empty()) {
    std::string combinedAssets =
        canonicalProfile + "\nshared_microbenchmark_sha256=" +
        profile.microbenchmarkContentSha256;
    llvm::ArrayRef<uint8_t> combinedBytes(
        reinterpret_cast<const uint8_t *>(combinedAssets.data()),
        combinedAssets.size());
    auto combinedHash = llvm::SHA256::hash(combinedBytes);
    profile.contentSha256 =
        llvm::toHex(llvm::ArrayRef<uint8_t>(combinedHash), true);
  }
  return profile;
}

static int64_t mapValue(const llvm::StringMap<int64_t> &values,
                        llvm::StringRef key, int64_t fallback = 0) {
  auto iterator = values.find(key);
  return iterator == values.end() ? fallback : iterator->second;
}

static std::vector<std::pair<llvm::StringRef, int64_t>>
getProfileOpElements(const SimdSimtFeatureSummary &features) {
  const int64_t maxNumel = std::max<int64_t>(1, features.maxTensorNumel);
  auto work = [&](llvm::StringRef elementName, int64_t rawCount) {
    auto iterator = features.opElements.find(elementName);
    if (iterator != features.opElements.end())
      return std::max<int64_t>(0, iterator->second);
    return std::max<int64_t>(0, rawCount) * maxNumel;
  };
  return {
      {"f32.add", work("add", features.addOps)},
      {"f32.sub", work("sub", features.subOps)},
      {"f32.mul", work("mul", features.mulOps)},
      {"f32.div", work("div", features.divOps)},
      {"f32.max", work("max", features.maxOps)},
      {"f32.abs", work("abs", features.absOps)},
      {"f32.exp", work("exp", features.expOps)},
      {"f32.log", work("log", features.logOps)},
      {"predicate.cmp", work("cmp", features.cmpOps)},
      {"predicate.select", work("select", features.selectOps)},
      {"convert.cast", work("cast", features.castOps)},
      {"f32.clamp", work("clamp", features.clampOps)},
  };
}

static std::pair<bool, std::string>
rankingCalibrationCoverage(const SimdSimtFeatureSummary &features,
                           int64_t weightedReductions, int64_t dotFlops,
                           const CandidateProfile &profile,
                           double irregularDensity) {
  const CoverageProfile &coverage = profile.coverage;
  const int64_t maxNumel = features.maxTensorNumel;
  const int64_t staticLoopTrips = features.staticLoopTripCountSum;
  const int64_t maskRankSum = features.maskRankSum;
  if (dotFlops > 0 && dotFlops <= coverage.tinyDotFlopsMax &&
      staticLoopTrips == 0 &&
      maxNumel <= coverage.tinyDotMaxTensorNumel &&
      irregularDensity >= coverage.minimumIrregularDensity)
    return {true, "tiny_irregular_dot"};
  if (dotFlops == 0 && features.rank1IndirectVectorReduce &&
      weightedReductions > 0 &&
      weightedReductions <= coverage.rank1WeightedReductionsMax &&
      maxNumel <= coverage.rank1MaxTensorNumel &&
      staticLoopTrips <= coverage.rowwiseLoopTripSumMax)
    return {true, "rank1_indirect_vector_reduction"};
  if (dotFlops == 0 && staticLoopTrips > 0 &&
      staticLoopTrips <= coverage.rowwiseLoopTripSumMax && maskRankSum > 0 &&
      maskRankSum <= coverage.rowwiseMaskRankSumMax &&
      weightedReductions > 0 &&
      weightedReductions <= coverage.rowwiseWeightedReductionsMax &&
      maxNumel <= coverage.rowwiseMaxTensorNumel &&
      irregularDensity >= coverage.minimumIrregularDensity)
    return {true, "masked_rowwise_reduction"};
  return {false, "out_of_calibration_domain"};
}

static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores) {
  SimdSimtCandidateKind best = SimdSimtCandidateKind::AllSIMD;
  double bestScore = scores.allSimd;
  if (scores.allSimtOnly < bestScore) {
    best = SimdSimtCandidateKind::AllSIMTOnly;
    bestScore = scores.allSimtOnly;
  }
  if (scores.mixedSimdSimt < bestScore)
    best = SimdSimtCandidateKind::MixedSIMDSIMT;
  return best;
}

static SimdSimtCandidateKind
chooseRunnerUp(const SimdSimtCandidateScores &scores,
               SimdSimtCandidateKind best) {
  std::array<std::pair<double, SimdSimtCandidateKind>, 3> candidates = {
      std::make_pair(scores.allSimd, SimdSimtCandidateKind::AllSIMD),
      std::make_pair(scores.allSimtOnly, SimdSimtCandidateKind::AllSIMTOnly),
      std::make_pair(scores.mixedSimdSimt,
                     SimdSimtCandidateKind::MixedSIMDSIMT)};
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto &lhs, const auto &rhs) {
                     return lhs.first < rhs.first;
                   });
  for (const auto &candidate : candidates)
    if (candidate.second != best)
      return candidate.second;
  return SimdSimtCandidateKind::AllSIMTOnly;
}

static void sortAndUnique(std::vector<std::string> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

} // namespace

llvm::StringRef
mlir::ascend::stringifySimdSimtCandidate(SimdSimtCandidateKind candidate) {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return kAllSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return kAllSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return kMixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

double
SimdSimtCandidateScores::get(SimdSimtCandidateKind candidate) const {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return allSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return allSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return mixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

llvm::json::Object SimdSimtCandidateScores::toJSON() const {
  llvm::json::Object result;
  result[kAllSimd] = allSimd;
  result[kAllSimtOnly] = allSimtOnly;
  result[kMixedSimdSimt] = mixedSimdSimt;
  return result;
}

llvm::json::Object SimdSimtFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["load_ops"] = loadOps;
  result["store_ops"] = storeOps;
  result["reduce_ops"] = reduceOps;
  result["scan_ops"] = scanOps;
  result["gather_ops"] = gatherOps;
  result["dot_ops"] = dotOps;
  result["atomic_ops"] = atomicOps;
  result["histogram_ops"] = histogramOps;
  result["broadcast_ops"] = broadcastOps;
  result["expand_dims_ops"] = expandDimsOps;
  result["splat_ops"] = splatOps;
  result["addptr_ops"] = addPtrOps;
  result["arith_ops"] = arithOps;
  result["math_ops"] = mathOps;
  result["add_ops"] = addOps;
  result["sub_ops"] = subOps;
  result["mul_ops"] = mulOps;
  result["div_ops"] = divOps;
  result["max_ops"] = maxOps;
  result["abs_ops"] = absOps;
  result["exp_ops"] = expOps;
  result["log_ops"] = logOps;
  result["cmp_ops"] = cmpOps;
  result["select_ops"] = selectOps;
  result["cast_ops"] = castOps;
  result["clamp_ops"] = clampOps;
  result["scalar_ops"] = scalarOps;
  result["max_tensor_rank"] = maxTensorRank;
  result["max_tensor_numel"] = maxTensorNumel;
  result["max_element_bits"] = maxElementBits;
  result["mask_tensor_ops"] = maskTensorOps;
  result["mask_rank_sum"] = maskRankSum;
  result["mask_broadcast_ops"] = maskBroadcastOps;
  result["pointer_tensor_ops"] = pointerTensorOps;
  result["pointer_unstructured_dims"] = pointerUnstructuredDims;
  result["lane_dependent_pointer_ops"] = laneDependentPointerOps;
  result["row_local_reduce_ops"] = rowLocalReduceOps;
  result["scalar_load_ops"] = scalarLoadOps;
  result["scalar_store_ops"] = scalarStoreOps;
  result["vector_ptr_splat_ops"] = vectorPtrSplatOps;
  result["vector_reduce_to_scalar_ops"] = vectorReduceToScalarOps;
  result["rank1_indirect_vector_reduce"] = rank1IndirectVectorReduce;
  result["weighted_ops"] = ::toJSON(weightedOps);
  result["op_elements"] = ::toJSON(opElements);
  result["load_bytes"] = loadBytes;
  result["store_bytes"] = storeBytes;
  result["load_warp_instructions"] = loadWarpInstructions;
  result["store_warp_instructions"] = storeWarpInstructions;
  result["dot_flops"] = dotFlops;
  result["dot_output_elements"] = dotOutputElements;
  llvm::json::Array dotShapes;
  for (const auto &shape : dotMNK)
    dotShapes.push_back(
        llvm::json::Array({shape[0], shape[1], shape[2]}));
  result["dot_mnk"] = std::move(dotShapes);
  result["static_loop_count"] = staticLoopCount;
  result["static_loop_trip_count_sum"] = staticLoopTripCountSum;
  result["static_loop_trip_count_max"] = staticLoopTripCountMax;
  result["has_dot"] = hasDot;
  result["has_gather"] = hasGather;
  result["has_atomic"] = hasAtomic;
  result["has_histogram"] = hasHistogram;
  result["has_scan"] = hasScan;
  result["has_explicit_scope"] = hasExplicitScope;
  result["has_control_flow"] = hasControlFlow;
  llvm::json::Array mixedKinds;
  for (const std::string &kind : observedMixedKinds)
    mixedKinds.push_back(kind);
  result["observed_mixed_kinds"] = std::move(mixedKinds);
  result["mixed_required"] = !observedMixedKinds.empty();
  result["mandatory_mixed_enabled"] = false;
  return result;
}

llvm::json::Object SimdSimtCostBreakdown::toJSON(
    const SimdSimtFeatureSummary &features) const {
  llvm::json::Object result;
  llvm::json::Object compute;
  compute["simd"] = simdComputeCycles;
  compute["simt"] = simtComputeCycles;
  compute["simd_dot"] = simdDotCycles;
  compute["simt_dot"] = simtDotCycles;
  result["compute_only"] = std::move(compute);

  llvm::json::Object memory;
  memory["load_bytes"] = features.loadBytes;
  memory["store_bytes"] = features.storeBytes;
  memory["simd_load_system_cycles"] = simdLoadCycles;
  memory["simd_store_system_cycles"] = simdStoreCycles;
  memory["simd_roofline_system_cycles"] = simdMemoryCycles;
  memory["simt_load_warp_instructions"] = features.loadWarpInstructions;
  memory["simt_store_warp_instructions"] = features.storeWarpInstructions;
  memory["simt_load_system_cycles"] = simtLoadCycles;
  memory["simt_store_system_cycles"] = simtStoreCycles;
  memory["simt_roofline_system_cycles"] = simtMemoryCycles;
  result["memory"] = std::move(memory);

  llvm::json::Object structure;
  structure["irregular_density"] = irregularDensity;
  structure["tiny_dot_underfill"] = tinyDotUnderfill;
  structure["components"] = ::toJSON(structuralComponents);
  structure["penalty_ratio"] = structuralPenaltyRatio;
  structure["simd_relative_floor_system_cycles"] = structuralFloorCycles;
  result["structure"] = std::move(structure);

  llvm::json::Object mixed;
  mixed["simd_fraction"] = mixedSimdFraction;
  mixed["cost_source"] = mixedCostSource;
  mixed["tiny_dot_residual_ratio"] = tinyDotMixedResidualRatio;
  mixed["measured_num_warps"] = measuredNumWarps;
  mixed["complete_setup_system_cycles"] = mixedSetupCycles;
  mixed["standalone_setup_system_cycles"] = standaloneSimtSetupCycles;
  mixed["incremental_transition_system_cycles"] = transitionDeltaCycles;
  result["mixed"] = std::move(mixed);

  llvm::json::Object execution;
  execution["shuffle_warp_instructions"] = simtShuffleInstructions;
  execution["shuffle_system_cycles"] = simtShuffleCycles;
  execution["predicate_warp_instructions"] = simtPredicateInstructions;
  execution["predicate_system_cycles"] = simtPredicateCycles;
  execution["program_issue_scale"] = programIssueScale;
  execution["simd_setup_system_cycles"] = simdSetupCycles;
  execution["simt_setup_system_cycles"] = simtSetupCycles;
  execution["simd_issue_payload_system_cycles"] = simdIssuePayloadCycles;
  execution["simt_issue_payload_system_cycles"] = simtIssuePayloadCycles;
  result["simt_execution"] = std::move(execution);

  llvm::json::Object opBreakdown;
  opBreakdown["simd_ops_system_cycles"] = ::toJSON(simdOpSystemCycles);
  opBreakdown["simt_ops_system_cycles"] = ::toJSON(simtOpSystemCycles);
  result["op_breakdown"] = std::move(opBreakdown);
  return result;
}

llvm::json::Object SimdSimtCostReport::toJSON() const {
  llvm::json::Object result;
  result["schema_version"] = schemaVersion;
  result["model"] = model;
  result["profile_version"] = profileVersion;
  result["profile_target"] = profileTarget;
  result["actual_target"] = actualTarget;
  result["target_compatible"] = targetCompatible;
  result["profile_content_sha256"] = profileContentSha256;
  result["selection_profile_content_sha256"] =
      selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = microbenchmarkProfileVersion;
  sharedEvidence["target"] = microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] =
      microbenchmarkProfileContentSha256;
  result["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  result["unit"] = scoreUnit;
  result["score_scope"] = scoreScope;
  result["selection_score_valid"] = selectionScoreValid;
  result["absolute_cost_valid"] = absoluteCostValid;
  result["excludes"] =
      llvm::json::Array({"host_launch", "grid_wave_count"});
  result["candidate_costs_evaluated"] = candidateCostsEvaluated;
  if (candidateCostsEvaluated) {
    result["candidate_costs"] = candidateCosts.toJSON();
    result["candidate_ratios_to_best"] = candidateRatiosToBest.toJSON();
    result["decision_kind"] = stringifySimdSimtCandidate(decision);
    result["runner_up_kind"] = stringifySimdSimtCandidate(runnerUp);
    result["best_score"] = bestScore;
    result["runner_up_score"] = runnerUpScore;
    result["gain_score"] = gainScore;
    result["decision_advantage"] = decisionAdvantage;
    result["required_gain_score"] = requiredGainScore;
  } else {
    result["candidate_costs"] = nullptr;
    result["candidate_ratios_to_best"] = nullptr;
    result["decision_kind"] = nullptr;
    result["runner_up_kind"] = nullptr;
    result["best_score"] = nullptr;
    result["runner_up_score"] = nullptr;
    result["gain_score"] = nullptr;
    result["decision_advantage"] = nullptr;
    result["required_gain_score"] = nullptr;
  }
  result["selectable_candidates"] =
      llvm::json::Array({kAllSimd, kAllSimtOnly, kMixedSimdSimt});
  result["margin_ratio"] = marginRatio;
  result["ranking_confidence"] = rankingConfidence;
  result["minimum_confidence_for_decision"] =
      minimumConfidenceForDecision;
  result["absolute_confidence"] = absoluteConfidence;
  result["confidence"] = rankingConfidence;
  result["gate_passed"] = gatePassed;

  llvm::json::Array reasons;
  for (const std::string &reason : gateReasons)
    reasons.push_back(reason);
  result["gate_reasons"] = std::move(reasons);
  llvm::json::Array unsupportedValues;
  for (const std::string &value : unsupported)
    unsupportedValues.push_back(value);
  result["unsupported"] = std::move(unsupportedValues);

  llvm::json::Object structure;
  structure["calibration_covered"] = calibrationCovered;
  structure["calibration_domain"] = calibrationDomain;
  result["calibration"] = std::move(structure);

  llvm::json::Object contract;
  contract["version"] = 2;
  contract["enabled"] = false;
  contract["requested"] = !features.observedMixedKinds.empty();
  contract["mandatory_override_suppressed"] =
      !features.observedMixedKinds.empty();
  contract["mandatory"] = false;
  contract["required"] = false;
  contract["target_kind"] = nullptr;
  llvm::json::Array routeKinds;
  for (const std::string &kind : features.observedMixedKinds)
    routeKinds.push_back(kind);
  contract["route_kinds"] = std::move(routeKinds);
  contract["all_simt_only_reference_only"] = false;
  result["mixed_execution_contract"] = std::move(contract);

  llvm::json::Object roles;
  roles[kAllSimd] = "selectable_candidate";
  roles[kAllSimtOnly] = "selectable_candidate";
  roles[kMixedSimdSimt] = "selectable_candidate";
  result["candidate_roles"] = std::move(roles);

  if (candidateCostsEvaluated) {
    llvm::json::Object analytical;
    analytical[kAllSimd] = breakdown.simdAnalyticalCycles;
    analytical[kAllSimtOnly] = breakdown.simtAnalyticalCycles;
    result["analytical_candidate_costs"] = std::move(analytical);

    llvm::json::Object detail = breakdown.toJSON(features);
    for (auto &entry : detail)
      result[entry.first] = std::move(entry.second);
    if (auto *structureObject = result.getObject("structure")) {
      (*structureObject)["calibration_covered"] = calibrationCovered;
      (*structureObject)["calibration_domain"] = calibrationDomain;
    }
  } else {
    result["analytical_candidate_costs"] = nullptr;
    llvm::json::Object structure;
    structure["irregular_density"] = breakdown.irregularDensity;
    structure["calibration_covered"] = calibrationCovered;
    structure["calibration_domain"] = calibrationDomain;
    result["structure"] = std::move(structure);
  }
  if (includeFeaturesInJSON)
    result["features"] = features.toJSON();
  result["des_feedback_applied"] = llvm::json::Array();
  result["des_feedback_validation_errors"] = llvm::json::Array();
  return result;
}

void SimdSimtCostReport::printJSON(llvm::raw_ostream &os, bool pretty) const {
  llvm::json::Object object = toJSON();
  if (pretty)
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(object)));
  else
    os << llvm::json::Value(std::move(object));
}

std::string mlir::ascend::getDefaultSimdSimtProfilePath() {
  if (const char *environment =
          std::getenv("TRITON_ASCEND_SIMD_SIMT_PROFILE"))
    if (*environment)
      return environment;
#ifdef TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH
  return TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH;
#else
  return {};
#endif
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");

  SimdSimtFeatureSummary features;
  initializeWorkMaps(features);
  llvm::SmallVector<Operation *> scans;

  auto updateTypeStats = [&](Type type) {
    features.maxElementBits =
        std::max(features.maxElementBits, getTypeBitWidth(type));
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      features.maxTensorRank =
          std::max<int64_t>(features.maxTensorRank, tensor.getRank());
      features.maxTensorNumel =
          std::max(features.maxTensorNumel, getStaticNumElements(type));
    }
  };

  module.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    const int64_t elements = getOperationElements(op);
    const int64_t loopMultiplier = getLoopMultiplier(op);

    for (Type type : op->getOperandTypes())
      updateTypeStats(type);
    for (Type type : op->getResultTypes())
      updateTypeStats(type);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          updateTypeStats(argument.getType());

    if (name.starts_with("arith."))
      ++features.arithOps;
    if (name.starts_with("math."))
      ++features.mathOps;
    if (name.starts_with("scf.") || name.starts_with("cf."))
      features.hasControlFlow = true;
    if (name == "scope.scope")
      features.hasExplicitScope = true;

    auto incrementRaw = [&](int64_t &counter) { ++counter; };
    if (name == "tt.load")
      incrementRaw(features.loadOps);
    else if (name == "tt.store")
      incrementRaw(features.storeOps);
    else if (name == "tt.reduce")
      incrementRaw(features.reduceOps);
    else if (name == "tt.scan" || name == "tt.associative_scan") {
      incrementRaw(features.scanOps);
      scans.push_back(op);
    } else if (name == "tt.gather")
      incrementRaw(features.gatherOps);
    else if (name == "tt.dot")
      incrementRaw(features.dotOps);
    else if (name.starts_with("tt.atomic"))
      incrementRaw(features.atomicOps);
    else if (name == "tt.histogram")
      incrementRaw(features.histogramOps);
    else if (name == "tt.broadcast")
      incrementRaw(features.broadcastOps);
    else if (name == "tt.expand_dims")
      incrementRaw(features.expandDimsOps);
    else if (name == "tt.splat")
      incrementRaw(features.splatOps);
    else if (name == "tt.addptr")
      incrementRaw(features.addPtrOps);

    if (name == "arith.addf" || name == "arith.addi")
      incrementRaw(features.addOps);
    else if (name == "arith.subf" || name == "arith.subi")
      incrementRaw(features.subOps);
    else if (name == "arith.mulf" || name == "arith.muli")
      incrementRaw(features.mulOps);
    else if (name == "arith.divf" || name == "arith.divsi" ||
             name == "arith.divui")
      incrementRaw(features.divOps);
    else if (name == "arith.maxnumf" || name == "arith.maxf" ||
             name == "arith.maxsi" || name == "arith.maxui")
      incrementRaw(features.maxOps);
    else if (name == "math.absf" || name == "math.absi")
      incrementRaw(features.absOps);
    else if (name == "math.exp")
      incrementRaw(features.expOps);
    else if (name == "math.log")
      incrementRaw(features.logOps);
    else if (name == "arith.cmpf" || name == "arith.cmpi")
      incrementRaw(features.cmpOps);
    else if (name == "arith.select")
      incrementRaw(features.selectOps);
    else if (isCastOp(name))
      incrementRaw(features.castOps);
    else if (name.starts_with("tt.clamp"))
      incrementRaw(features.clampOps);

    llvm::StringRef weightedKind = classifyWeightedOp(name);
    if (!weightedKind.empty()) {
      features.weightedOps[weightedKind] += loopMultiplier;
      features.opElements[weightedKind] += elements * loopMultiplier;
    }

    if (name == "scf.for") {
      int64_t tripCount = getStaticLoopTripCount(op);
      ++features.staticLoopCount;
      features.staticLoopTripCountSum += tripCount;
      features.staticLoopTripCountMax =
          std::max(features.staticLoopTripCountMax, tripCount);
    }

    auto dataTypeAndElements = [&](bool load)
        -> std::pair<Type, int64_t> {
      if (load && op->getNumResults() > 0)
        return {op->getResult(0).getType(),
                getStaticNumElements(op->getResult(0).getType())};
      if (!load && op->getNumOperands() > 1)
        return {op->getOperand(1).getType(),
                getStaticNumElements(op->getOperand(1).getType())};
      if (op->getNumOperands() > 0)
        return {op->getOperand(0).getType(), elements};
      return {Type(), elements};
    };
    if (name == "tt.load" || name == "tt.store") {
      bool load = name == "tt.load";
      auto [dataType, dataElements] = dataTypeAndElements(load);
      int64_t bitWidth = dataType ? getTypeBitWidth(dataType) : 32;
      double bytes =
          static_cast<double>(dataElements) * loopMultiplier * bitWidth / 8.0;
      int64_t warpInstructions =
          static_cast<int64_t>(std::ceil(dataElements / 32.0)) *
          loopMultiplier;
      if (load) {
        features.loadBytes += bytes;
        features.loadWarpInstructions += warpInstructions;
      } else {
        features.storeBytes += bytes;
        features.storeWarpInstructions += warpInstructions;
      }
    }

    if (name == "tt.dot" && op->getNumOperands() >= 2) {
      auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      if (lhs && rhs && lhs.getRank() >= 2 && rhs.getRank() >= 2) {
        int64_t m = lhs.getShape()[lhs.getRank() - 2];
        int64_t k = lhs.getShape()[lhs.getRank() - 1];
        int64_t n = rhs.getShape()[rhs.getRank() - 1];
        if (m > 0 && n > 0 && k > 0) {
          features.dotFlops += 2 * m * n * k * loopMultiplier;
          features.dotOutputElements += m * n * loopMultiplier;
          features.dotMNK.push_back({m, n, k});
        }
      }
    }

    std::vector<int64_t> rankedResultAndOperandRanks;
    bool hasRankedInput = false;
    bool hasRankedResult = false;
    for (Type type : op->getOperandTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        rankedResultAndOperandRanks.push_back(tensor.getRank());
        hasRankedInput = true;
      }
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        rankedResultAndOperandRanks.push_back(tensor.getRank());
        hasRankedResult = true;
      }
    if (name == "tt.reduce") {
      if (rankedResultAndOperandRanks.size() > 1) {
        auto [minimum, maximum] =
            std::minmax_element(rankedResultAndOperandRanks.begin(),
                                rankedResultAndOperandRanks.end());
        if (*maximum > *minimum)
          ++features.rowLocalReduceOps;
      }
      if (hasRankedInput && !hasRankedResult)
        ++features.vectorReduceToScalarOps;
    }

    std::vector<int64_t> maskRanks;
    for (Type type : op->getOperandTypes())
      if (isMaskTensorType(type))
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());
    for (Type type : op->getResultTypes())
      if (isMaskTensorType(type))
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());
    if (!maskRanks.empty()) {
      ++features.maskTensorOps;
      for (int64_t rank : maskRanks)
        features.maskRankSum += rank;
      if (name == "tt.broadcast" || name == "tt.expand_dims")
        ++features.maskBroadcastOps;
    }

    bool isPointerOperation =
        name == "tt.addptr" || name == "tt.load" || name == "tt.store";
    if (isPointerOperation) {
      std::set<std::string> uniqueShapes;
      auto collectShape = [&](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor)
          return;
        std::string key;
        llvm::raw_string_ostream os(key);
        os << tensor.getRank();
        for (int64_t dim : tensor.getShape())
          os << 'x' << dim;
        os.flush();
        uniqueShapes.insert(std::move(key));
      };
      for (Type type : op->getOperandTypes())
        collectShape(type);
      for (Type type : op->getResultTypes())
        collectShape(type);
      int64_t maxPointerRank = 0;
      for (const std::string &shape : uniqueShapes) {
        llvm::StringRef shapeRef(shape);
        int64_t rank = 0;
        (void)shapeRef.take_front(shapeRef.find('x')).getAsInteger(10, rank);
        ++features.pointerTensorOps;
        maxPointerRank = std::max(maxPointerRank, rank);
        if (rank > 1)
          features.pointerUnstructuredDims += rank;
      }
      if (maxPointerRank > 1)
        ++features.laneDependentPointerOps;
    }

    bool anyRankedType = llvm::any_of(
        op->getOperandTypes(),
        [](Type type) { return isa<RankedTensorType>(type); });
    anyRankedType |= llvm::any_of(
        op->getResultTypes(),
        [](Type type) { return isa<RankedTensorType>(type); });
    if (name == "tt.load" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarLoadOps;
    if (name == "tt.store" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarStoreOps;
    if (name == "tt.splat" && op->getNumOperands() > 0 &&
        op->getNumResults() > 0 &&
        isPointerType(op->getOperand(0).getType()) &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      ++features.vectorPtrSplatOps;
  });

  features.scalarOps =
      features.addOps + features.subOps + features.mulOps +
      features.divOps + features.maxOps + features.absOps +
      features.expOps + features.logOps + features.cmpOps +
      features.selectOps + features.castOps + features.clampOps;
  features.hasDot = features.dotOps > 0;
  features.hasGather = features.gatherOps > 0;
  features.hasAtomic = features.atomicOps > 0;
  features.hasHistogram = features.histogramOps > 0;
  features.hasScan = features.scanOps > 0;
  features.rank1IndirectVectorReduce =
      features.maxTensorRank == 1 && features.reduceOps > 0 &&
      features.vectorReduceToScalarOps > 0 &&
      features.vectorPtrSplatOps > 0 && features.scalarLoadOps >= 2;

  if (features.gatherOps > 0)
    appendUnique(features.observedMixedKinds, "direct_gather");
  if (features.histogramOps > 0)
    appendUnique(features.observedMixedKinds, "histogram");
  for (Operation *scan : scans)
    if (isPlainOneDimensionalCumsum(scan))
      appendUnique(features.observedMixedKinds, "plain_1d_cumsum");
  bool indirectMixedCandidate =
      features.rank1IndirectVectorReduce ||
      (features.laneDependentPointerOps > 0 &&
       (features.maskBroadcastOps > 0 || features.staticLoopCount > 0 ||
        (features.dotOps > 0 && features.loadOps >= 3)));
  if (indirectMixedCandidate)
    appendUnique(features.observedMixedKinds,
                 "conditional_indirect_memory");
  if (features.atomicOps > 0)
    appendUnique(features.observedMixedKinds,
                 "conditional_indirect_atomic");
  return features;
}

llvm::Expected<SimdSimtCostReport>
mlir::ascend::estimateSimdSimtCandidates(
    const SimdSimtFeatureSummary &features,
    const SimdSimtCostModelOptions &options) {
  if (!std::isfinite(options.marginRatio) || options.marginRatio < 0.0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT marginRatio must be finite and non-negative");
  auto profileOrError = loadCandidateProfile(options.profilePath);
  if (!profileOrError)
    return profileOrError.takeError();
  CandidateProfile profile = std::move(*profileOrError);

  SimdSimtCostReport report;
  report.profileVersion = profile.profileVersion;
  report.profileTarget = profile.target;
  report.actualTarget = options.actualTarget;
  report.profileContentSha256 = profile.contentSha256;
  report.selectionProfileContentSha256 =
      profile.selectionContentSha256;
  report.microbenchmarkProfileVersion =
      profile.microbenchmarkProfileVersion;
  report.microbenchmarkProfileTarget =
      profile.microbenchmarkProfileTarget;
  report.microbenchmarkProfileContentSha256 =
      profile.microbenchmarkContentSha256;
  report.scoreUnit = profile.scoreUnit;
  report.minimumConfidenceForDecision = profile.minimumConfidence;
  report.targetCompatible =
      targetMatches(profile, options.actualTarget);
  report.features = features;
  report.includeFeaturesInJSON = options.includeFeaturesInJSON;
  report.marginRatio = options.marginRatio;

  // Coverage is a validity check over extracted features, not a cost term.
  // Evaluate it before all resource, structural, and transition scoring so
  // production auto selection can reject out-of-domain kernels cheaply.
  const int64_t weightedReductions =
      mapValue(features.weightedOps, "reduce", features.reduceOps);
  const int64_t dotFlops = features.dotFlops;
  const int64_t pointerOps =
      std::max<int64_t>(1, features.pointerTensorOps);
  report.breakdown.irregularDensity =
      std::min(1.0, static_cast<double>(features.laneDependentPointerOps) /
                        pointerOps);
  auto [covered, domain] = rankingCalibrationCoverage(
      features, weightedReductions, dotFlops, profile,
      report.breakdown.irregularDensity);
  report.calibrationCovered = covered;
  report.calibrationDomain = std::move(domain);
  report.selectionScoreValid = covered;

  if (!report.calibrationCovered &&
      !options.scoreOutsideCalibrationCoverage) {
    if (!report.targetCompatible)
      report.gateReasons.push_back("target_incompatible");
    report.gateReasons.push_back("selection_score_invalid");
    report.gatePassed = false;
    return report;
  }

  const int64_t numWarps =
      std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
  const int64_t maxNumel = std::max<int64_t>(1, features.maxTensorNumel);
  const int64_t elementBits = std::max<int64_t>(8, features.maxElementBits);
  const int64_t vectorWidth =
      std::max<int64_t>(1, profile.simdVectorWidthBits / elementBits);
  std::vector<std::string> resourceConfidence;

  llvm::StringMap<int64_t> rawCountByKind;
  rawCountByKind["gather"] = features.gatherOps;
  rawCountByKind["histogram"] = features.histogramOps;
  rawCountByKind["atomic"] = features.atomicOps;
  for (llvm::StringRef kind : {"gather", "histogram", "atomic"}) {
    int64_t coreWork =
        mapValue(features.opElements, kind,
                 mapValue(rawCountByKind, kind) * maxNumel);
    if (coreWork > 0)
      report.unsupported.push_back(
          (kind + "_core_cost_uncalibrated").str());
  }

  int64_t classifiedScalarOps =
      features.addOps + features.subOps + features.mulOps +
      features.divOps + features.maxOps + features.absOps +
      features.expOps + features.logOps + features.cmpOps +
      features.selectOps + features.castOps + features.clampOps;
  int64_t unclassifiedScalarOps =
      std::max<int64_t>(0, features.scalarOps - classifiedScalarOps);
  if (unclassifiedScalarOps)
    report.unsupported.push_back(std::to_string(unclassifiedScalarOps) +
                                 " unclassified arithmetic ops");

  for (const auto &[opName, elements] : getProfileOpElements(features)) {
    if (elements <= 0)
      continue;
    auto simdIterator = profile.simdOps.find(opName);
    auto simtIterator = profile.simtOps.find(opName);
    if (simdIterator == profile.simdOps.end() ||
        simtIterator == profile.simtOps.end()) {
      report.unsupported.push_back(opName.str());
      continue;
    }
    const OpProfile &simd = simdIterator->second;
    const OpProfile &simt = simtIterator->second;
    if (simd.throughput <= 0.0 || simt.throughput <= 0.0) {
      report.unsupported.push_back(opName.str());
      continue;
    }
    double simdCycles =
        std::ceil(static_cast<double>(elements) / vectorWidth) /
        simd.throughput * simd.factor;
    double simtCycles =
        static_cast<double>(elements) / simt.throughput * simt.factor;
    report.breakdown.simdOpSystemCycles[opName] = simdCycles;
    report.breakdown.simtOpSystemCycles[opName] = simtCycles;
    report.breakdown.simdComputeCycles += simdCycles;
    report.breakdown.simtComputeCycles += simtCycles;
    resourceConfidence.push_back(simd.confidence);
    resourceConfidence.push_back(simt.confidence);
  }

  report.breakdown.simdLoadCycles =
      features.loadBytes / profile.simdMte2BytesPerCycle;
  report.breakdown.simdStoreCycles =
      features.storeBytes / profile.simdMte3BytesPerCycle;
  report.breakdown.simdMemoryCycles =
      std::max(report.breakdown.simdLoadCycles,
               report.breakdown.simdStoreCycles);
  if (features.loadBytes != 0.0 || features.storeBytes != 0.0)
    resourceConfidence.push_back(profile.simdMemoryConfidence);

  const int64_t loadWarpInstructions =
      features.loadWarpInstructions != 0
          ? features.loadWarpInstructions
          : features.loadOps *
                static_cast<int64_t>(std::ceil(
                    static_cast<double>(maxNumel) / profile.simtWarpSize));
  const int64_t storeWarpInstructions =
      features.storeWarpInstructions != 0
          ? features.storeWarpInstructions
          : features.storeOps *
                static_cast<int64_t>(std::ceil(
                    static_cast<double>(maxNumel) / profile.simtWarpSize));
  report.features.loadWarpInstructions = loadWarpInstructions;
  report.features.storeWarpInstructions = storeWarpInstructions;
  report.breakdown.simtLoadCycles =
      loadWarpInstructions / profile.simtLoadWarpRate;
  report.breakdown.simtStoreCycles =
      storeWarpInstructions / profile.simtStoreWarpRate;
  report.breakdown.simtMemoryCycles =
      report.breakdown.simtLoadCycles + report.breakdown.simtStoreCycles;
  if (loadWarpInstructions != 0 || storeWarpInstructions != 0)
    resourceConfidence.push_back(profile.simtMemoryConfidence);

  const int64_t weightedScans =
      mapValue(features.weightedOps, "scan", features.scanOps);
  if (weightedScans)
    report.unsupported.push_back("scan_template_ranking_uncalibrated");
  const int64_t shuffleLevels = static_cast<int64_t>(
      std::ceil(std::log2(static_cast<double>(profile.simtWarpSize))));
  report.breakdown.simtShuffleInstructions =
      static_cast<double>(weightedReductions + weightedScans) *
      std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize) *
      shuffleLevels;
  report.breakdown.simtShuffleCycles =
      report.breakdown.simtShuffleInstructions / profile.simtShuffleRate;
  if (report.breakdown.simtShuffleInstructions != 0.0)
    resourceConfidence.push_back(profile.simtShuffleConfidence);

  report.breakdown.simtPredicateInstructions =
      static_cast<double>(features.maskRankSum) *
      std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize);
  report.breakdown.simtPredicateCycles =
      report.breakdown.simtPredicateInstructions /
      profile.simtPredicateRate;

  if (dotFlops) {
    report.breakdown.simdDotCycles =
        profile.simdDotSetupCycles +
        static_cast<double>(dotFlops) / profile.simdDotFlopsPerCycle;
    report.breakdown.simtDotCycles =
        profile.simtDotSetupCycles +
        static_cast<double>(dotFlops) / profile.simtDotFlopsPerCycle;
    resourceConfidence.push_back(profile.simdDotConfidence);
    resourceConfidence.push_back(profile.simtDotConfidence);
  }

  report.breakdown.simdSetupCycles = profile.simdSetupCycles;
  report.breakdown.simtSetupCycles = profile.simtSetupCycles;
  report.breakdown.simdIssuePayloadCycles =
      std::max(report.breakdown.simdComputeCycles +
                   report.breakdown.simdDotCycles,
               report.breakdown.simdMemoryCycles);
  report.breakdown.simtIssuePayloadCycles =
      std::max(report.breakdown.simtComputeCycles +
                   report.breakdown.simtShuffleCycles +
                   report.breakdown.simtDotCycles,
               report.breakdown.simtMemoryCycles) +
      report.breakdown.simtPredicateCycles;
  report.breakdown.programIssueScale = profile.programIssueScale;
  report.breakdown.simdAnalyticalCycles =
      profile.simdSetupCycles +
      report.breakdown.simdIssuePayloadCycles * profile.programIssueScale;
  report.breakdown.simtAnalyticalCycles =
      profile.simtSetupCycles +
      report.breakdown.simtIssuePayloadCycles * profile.programIssueScale;

  const bool tinyDot =
      dotFlops > 0 && dotFlops <= profile.structural.tinyDotFlopsMax;
  report.breakdown.tinyDotUnderfill =
      tinyDot ? std::max(0.0, 1.0 -
                                 static_cast<double>(dotFlops) /
                                     profile.structural.tinyDotFlopsMax)
              : 0.0;
  const double irregularPerDensity =
      tinyDot ? profile.structural.tinyDotIrregularPerDensity
              : profile.structural.irregularPerDensity;
  const double irregularCap =
      tinyDot ? profile.structural.tinyDotIrregularCap
              : profile.structural.irregularCap;
  report.breakdown.structuralComponents["irregular_addressing"] =
      std::min(irregularCap,
               report.breakdown.irregularDensity * irregularPerDensity);
  report.breakdown.structuralComponents["mask_materialization"] =
      std::min(profile.structural.maskCap,
               features.maskRankSum * profile.structural.perMaskRank);
  report.breakdown.structuralComponents["reduction_lowering"] =
      std::min(profile.structural.reductionCap,
               weightedReductions *
                   profile.structural.perWeightedReduction);
  report.breakdown.structuralComponents["static_loop_control"] =
      std::min(profile.structural.loopCap,
               features.staticLoopTripCountSum *
                   profile.structural.perStaticLoopTrip);
  report.breakdown.structuralComponents["control_flow"] =
      features.hasControlFlow ? profile.structural.controlFlow : 0.0;
  report.breakdown.structuralComponents["tiny_dot_startup"] =
      tinyDot ? profile.structural.tinyDot *
                    report.breakdown.tinyDotUnderfill
              : 0.0;
  report.breakdown
      .structuralComponents["rank1_indirect_vector_reduction"] =
      features.rank1IndirectVectorReduce
          ? profile.structural.rank1IndirectVectorReduction
          : 0.0;
  for (const auto &component : report.breakdown.structuralComponents)
    report.breakdown.structuralPenaltyRatio += component.second;
  report.breakdown.structuralFloorCycles =
      report.breakdown.structuralPenaltyRatio > 0.0
          ? report.breakdown.simtAnalyticalCycles *
                (1.0 + report.breakdown.structuralPenaltyRatio)
          : 0.0;
  report.candidateCosts.allSimd =
      std::max(report.breakdown.simdAnalyticalCycles,
               report.breakdown.structuralFloorCycles);
  report.candidateCosts.allSimtOnly =
      report.breakdown.simtAnalyticalCycles;

  const TransitionProfile *nearestTransition = nullptr;
  for (const TransitionProfile &transition : profile.transitions)
    if (!nearestTransition ||
        std::abs(transition.numWarps - numWarps) <
            std::abs(nearestTransition->numWarps - numWarps))
      nearestTransition = &transition;
  if (!nearestTransition)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile has no transition");
  report.breakdown.measuredNumWarps = nearestTransition->numWarps;
  report.breakdown.standaloneSimtSetupCycles = profile.simtSetupCycles;
  report.breakdown.mixedSetupCycles =
      nearestTransition->emptySimtSetupCycles;
  report.breakdown.transitionDeltaCycles =
      std::max(0.0, report.breakdown.mixedSetupCycles -
                        report.breakdown.standaloneSimtSetupCycles);

  double mixedBlend = profile.mixedBlend.base;
  mixedBlend +=
      std::min(profile.mixedBlend.loopCap,
               features.staticLoopTripCountSum *
                   profile.mixedBlend.perStaticLoopTrip);
  mixedBlend +=
      std::min(profile.mixedBlend.maskCap,
               features.maskBroadcastOps *
                   profile.mixedBlend.perMaskBroadcast);
  mixedBlend +=
      std::min(profile.mixedBlend.reductionCap,
               weightedReductions *
                   profile.mixedBlend.perWeightedReduction);
  if (features.hasControlFlow)
    mixedBlend += profile.mixedBlend.controlFlow;
  if (tinyDot)
    mixedBlend += profile.mixedBlend.tinyDot;
  mixedBlend =
      std::min(profile.mixedBlend.max, std::max(0.0, mixedBlend));
  report.breakdown.mixedSimdFraction = mixedBlend;

  double simdPayload =
      std::max(0.0, report.candidateCosts.allSimd -
                        profile.simdSetupCycles);
  double simtPayload =
      std::max(0.0, report.candidateCosts.allSimtOnly -
                        profile.simtSetupCycles);
  double mixedPayload =
      simtPayload + mixedBlend * (simdPayload - simtPayload);
  report.candidateCosts.mixedSimdSimt =
      report.breakdown.mixedSetupCycles + std::max(0.0, mixedPayload);
  report.breakdown.mixedCostSource =
      "payload_partition_plus_complete_setup";
  if (tinyDot) {
    report.breakdown.tinyDotMixedResidualRatio =
        profile.tinyDotMixedPenaltyAtZero *
        report.breakdown.tinyDotUnderfill;
    report.candidateCosts.mixedSimdSimt =
        report.candidateCosts.allSimtOnly *
        (1.0 + report.breakdown.tinyDotMixedResidualRatio);
    report.breakdown.mixedCostSource =
        "tiny_dot_event_relative_underfill_residual";
  }

  report.candidateCostsEvaluated = true;
  sortAndUnique(report.unsupported);
  report.decision = chooseBest(report.candidateCosts);
  report.runnerUp =
      chooseRunnerUp(report.candidateCosts, report.decision);
  report.bestScore = report.candidateCosts.get(report.decision);
  report.runnerUpScore = report.candidateCosts.get(report.runnerUp);
  report.decisionAdvantage =
      report.decision == SimdSimtCandidateKind::AllSIMD
          ? report.runnerUpScore - report.bestScore
          : report.candidateCosts.allSimd - report.bestScore;
  report.gainScore = report.decisionAdvantage;
  report.requiredGainScore =
      std::max(64.0, report.candidateCosts.allSimd * options.marginRatio);
  double ratioDenominator = std::max(
      1.0e-9,
      std::min({report.candidateCosts.allSimd,
                report.candidateCosts.allSimtOnly,
                report.candidateCosts.mixedSimdSimt}));
  report.candidateRatiosToBest = {
      report.candidateCosts.allSimd / ratioDenominator,
      report.candidateCosts.allSimtOnly / ratioDenominator,
      report.candidateCosts.mixedSimdSimt / ratioDenominator};

  report.absoluteConfidence = minimumConfidence(resourceConfidence);
  if (!report.unsupported.empty()) {
    report.rankingConfidence = "none";
  } else if (report.breakdown.structuralPenaltyRatio > 0.0 && covered) {
    report.rankingConfidence = minimumConfidence(
        {report.absoluteConfidence, profile.rankingConfidence,
         profile.transitionConfidence});
  } else {
    report.rankingConfidence = report.absoluteConfidence;
  }
  if (!report.targetCompatible)
    report.rankingConfidence = "none";

  if (!report.targetCompatible)
    report.gateReasons.push_back("target_incompatible");
  if (!report.selectionScoreValid)
    report.gateReasons.push_back("selection_score_invalid");
  if (!report.unsupported.empty())
    report.gateReasons.push_back("unsupported_cost_terms");
  if (confidenceRank(report.rankingConfidence) <
      confidenceRank(report.minimumConfidenceForDecision))
    report.gateReasons.push_back(
        "ranking_confidence_" + report.rankingConfidence + "_below_" +
        report.minimumConfidenceForDecision);
  if (!(report.decisionAdvantage > report.requiredGainScore))
    report.gateReasons.push_back(
        "decision_advantage_not_above_required_gain");
  report.gatePassed = report.gateReasons.empty();
  return report;
}

llvm::Expected<SimdSimtCostReport>
mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimdSimtCostModelOptions &options) {
  auto features = analyzeSimdSimtFeatures(module);
  if (!features)
    return features.takeError();
  return estimateSimdSimtCandidates(*features, options);
}
