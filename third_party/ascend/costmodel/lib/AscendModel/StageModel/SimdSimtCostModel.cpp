//===- SimdSimtCostModel.cpp - Ascend SIMD/SIMT candidate model ----------===//
//
// The numerical model in this file is the versioned C++ candidate model.  It
// intentionally produces a relative per-program selection score, not an
// end-to-end kernel-time prediction.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/StageModel/SimdSimtCostModel.h"
#include "AscendModel/Profile/MicrobenchmarkProfile.h"
#include "AscendModel/StageModel/SimtAnchorAnalysis.h"
#include "AscendModel/StageModel/StageCostModels.h"
#include "AscendModel/StageModel/StageDiscovery.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

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

struct StructuralProfile {
  int64_t tinyDotFlopsMax = 0;
};

struct StageResourceProfile {
  double scalarOperationsPerCycle = 0.0;
  double issueOperationsPerCycle = 0.0;
  double spillTransactionsPerCycle = 0.0;
  double indirectLoadTransactionsPerCycle = 0.0;
  double indirectStoreTransactionsPerCycle = 0.0;
  double indirectDependencyLatencyCycles = 0.0;
  StageControlFlowRates controlFlow;

  bool isValid() const {
    const std::array<double, 5> positive = {
        scalarOperationsPerCycle, issueOperationsPerCycle,
        spillTransactionsPerCycle, indirectLoadTransactionsPerCycle,
        indirectStoreTransactionsPerCycle};
    return llvm::all_of(positive,
                        [](double value) {
                          return std::isfinite(value) && value > 0.0;
                        }) &&
           std::isfinite(indirectDependencyLatencyCycles) &&
           indirectDependencyLatencyCycles >= 0.0 &&
           controlFlow.isFiniteAndNonNegative();
  }
};

struct CandidateProfile {
  std::string profileVersion;
  std::string target;
  std::vector<std::string> compatibleTargets;
  std::string scoreUnit;
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;

  StructuralProfile structural;
  int64_t simdVectorWidthBits = 2048;
  double simdSetupCycles = 0.0;
  llvm::StringMap<OpProfile> simdOps;
  double simdMte2BytesPerCycle = 0.0;
  double simdMte3BytesPerCycle = 0.0;
  std::string simdMemoryConfidence = "none";
  double simdDotSetupCycles = 0.0;
  double simdDotFlopsPerCycle = 0.0;
  std::string simdDotConfidence = "none";
  StageResourceProfile simdStageResources;

  int64_t simtWarpSize = 32;
  double simtSetupCycles = 0.0;
  std::string simtSetupConfidence = "none";
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
  StageResourceProfile simtStageResources;
  int64_t superblockUsefulFactorLimit = 1;
  int64_t superblockPersistentStatePressureFreeFactor = 1;
  double superblockPersistentStateBytesPerCycle = 0.0;
  double scopeHandoffFixedDirectionalCycles = 0.0;
  double scopeSimdUbLoadBytesPerCycle = 0.0;
  double scopeSimdUbStoreBytesPerCycle = 0.0;
  double scopeSimtUbLoadBytesPerCycle = 0.0;
  double scopeSimtUbStoreBytesPerCycle = 0.0;
  llvm::DenseMap<int64_t, double> scopeSetupProxyCycles;
  std::string scopeSetupProxyConfidence = "none";
  std::string scopeSetupProxySource;
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
                                 llvm::StringRef key, llvm::StringRef context) {
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

  int64_t optionalInteger(const llvm::json::Object &parent, llvm::StringRef key,
                          int64_t defaultValue) {
    if (auto value = parent.getInteger(key))
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

template <typename SummaryT>
static void initializeWorkMaps(SummaryT &features) {
  for (llvm::StringRef key :
       {"load", "store", "reduce", "scan", "gather", "histogram", "atomic",
        "add", "sub", "mul", "div", "max", "abs", "exp", "log", "cmp", "select",
        "cast", "clamp"}) {
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

/// Return true only when a loop-carried value is used exclusively to derive
/// load/store addresses (plus the loop yield).  This recognizes pointer and
/// integer-offset induction without confusing it with a recurrence whose
/// value feeds arithmetic, predicates, or stored data.
static bool isAddressOnlyLoopCarriedValue(Value root) {
  llvm::SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  bool reachesAddressUse = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      llvm::StringRef name = user->getName().getStringRef();
      if (name == "scf.yield")
        continue;
      if ((name == "tt.load" || name == "tt.store") &&
          use.getOperandNumber() == 0) {
        reachesAddressUse = true;
        continue;
      }
      const bool addressForwarding =
          name == "tt.addptr" || name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims" || name == "arith.addi" ||
          name == "arith.subi" || name == "arith.muli" ||
          name == "arith.index_cast";
      if (!addressForwarding)
        return false;
      if (name == "tt.addptr")
        reachesAddressUse = true;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return reachesAddressUse;
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

static double getStaticTensorBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return 0.0;
  return static_cast<double>(getStaticNumElements(type)) *
         getTypeBitWidth(tensor.getElementType()) / 8.0;
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

static std::optional<int64_t> getKnownStaticLoopTripCount(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumOperands() < 3)
    return std::nullopt;
  auto lower = getConstantInteger(op->getOperand(0));
  auto upper = getConstantInteger(op->getOperand(1));
  auto step = getConstantInteger(op->getOperand(2));
  if (!lower || !upper || !step || *step == 0)
    return std::nullopt;
  int64_t span = *upper - *lower;
  if (span > 0 && *step > 0)
    return std::max<int64_t>(1, (span + *step - 1) / *step);
  if (span < 0 && *step < 0) {
    int64_t positiveSpan = -span;
    int64_t positiveStep = -*step;
    return std::max<int64_t>(1,
                             (positiveSpan + positiveStep - 1) / positiveStep);
  }
  return std::nullopt;
}

static int64_t getModeledLoopTripCount(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  if (auto knownTripCount = getKnownStaticLoopTripCount(op))
    return *knownTripCount;
  if (auto iterator = structuralTripEstimates.find(op);
      iterator != structuralTripEstimates.end())
    return iterator->second;
  return 1;
}

static int64_t getLoopMultiplier(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  int64_t multiplier = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "scf.for")
      continue;
    // AutoBlockify V1 wraps the original logical program in a physical-core
    // scheduling loop.  Its dispatch cost is modeled as a separate phase;
    // it must not multiply every algorithm operation as if it were an
    // algorithmic loop.
    if (parent->hasAttr("ta.auto_blockify_v1.schedule"))
      continue;
    int64_t tripCount =
        getModeledLoopTripCount(parent, structuralTripEstimates);
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
  if (name == "arith.divf" || name == "arith.divsi" || name == "arith.divui")
    return "div";
  if (name == "arith.maxnumf" || name == "arith.maxf" ||
      name == "arith.maxsi" || name == "arith.maxui")
    return "max";
  if (name == "math.absf" || name == "math.absi")
    return "abs";
  if (name == "math.exp")
    return "exp";
  if (name == "math.log")
    return "log";
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
  return *std::min_element(values.begin(), values.end(),
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
    auto value =
        microbench->requireValue(*reference, expectedUnit, expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return 0.0;
    }
    if (measurementConfidence) {
      const MicrobenchmarkMeasurement *measurement =
          microbench->getMeasurement(*reference);
      *measurementConfidence = measurement ? measurement->confidence : "none";
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
  std::string path = requestedPath.empty() ? getDefaultSimdSimtProfilePath()
                                           : requestedPath.str();
  if (path.empty())
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "SIMD/SIMT profile path is empty; set "
                                   "TRITON_ASCEND_SIMD_SIMT_PROFILE");

  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read SIMD/SIMT profile '%s'",
                                   path.c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse SIMD/SIMT profile '%s': %s",
                                   path.c_str(),
                                   llvm::toString(parsed.takeError()).c_str());
  const auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile root must be an object");
  auto selectionSchemaVersion = root->getInteger("schema_version");
  if (!selectionSchemaVersion || *selectionSchemaVersion != 12)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageModel profile schema_version must be 12");

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

  profile.profileVersion = reader.string(*root, "profile_version", "profile");
  profile.target = reader.string(*root, "target", "profile");
  if (microbench && llvm::StringRef(profile.target) != microbench->getTarget())
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
  if (const auto *discovery =
          reader.object(*root, "stage_discovery", "profile"))
    profile.structural.tinyDotFlopsMax =
        reader.integer(*discovery, "tiny_dot_flops_max", "stage_discovery");

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    if (simd->getString("vector_width_measurement")) {
      profile.simdVectorWidthBits =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
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
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
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
    if (const auto *resources =
            reader.object(*simd, "stage_resources", "simd")) {
      profile.simdStageResources.scalarOperationsPerCycle =
          reader.number(*resources, "scalar_operations_per_system_cycle",
                        "simd.stage_resources");
      profile.simdStageResources.issueOperationsPerCycle =
          reader.number(*resources, "issue_instructions_per_system_cycle",
                        "simd.stage_resources");
      profile.simdStageResources.spillTransactionsPerCycle =
          reader.number(*resources, "spill_transactions_per_system_cycle",
                        "simd.stage_resources");
      if (const auto *indirect = reader.object(*resources, "indirect_memory",
                                               "simd.stage_resources")) {
        profile.simdStageResources.indirectLoadTransactionsPerCycle =
            reader.number(*indirect, "load_transactions_per_system_cycle",
                          "simd.stage_resources.indirect_memory");
        profile.simdStageResources.indirectStoreTransactionsPerCycle =
            reader.number(*indirect, "store_transactions_per_system_cycle",
                          "simd.stage_resources.indirect_memory");
        profile.simdStageResources.indirectDependencyLatencyCycles =
            reader.number(*indirect, "dependency_latency_system_cycles",
                          "simd.stage_resources.indirect_memory");
      }
      if (const auto *control = reader.object(*resources, "control_flow",
                                              "simd.stage_resources")) {
        profile.simdStageResources.controlFlow.loopBackedgeCycles =
            reader.number(*control, "loop_backedge_system_cycles",
                          "simd.stage_resources.control_flow");
        profile.simdStageResources.controlFlow.conditionalBranchCycles =
            reader.number(*control, "conditional_branch_system_cycles",
                          "simd.stage_resources.control_flow");
        profile.simdStageResources.controlFlow.divergentBranchPenaltyCycles =
            reader.number(*control, "divergent_branch_penalty_system_cycles",
                          "simd.stage_resources.control_flow");
        profile.simdStageResources.controlFlow.synchronizationCycles =
            reader.number(*control, "synchronization_system_cycles",
                          "simd.stage_resources.control_flow");
      }
    }
  }

  const auto *simt = reader.object(*root, "simt", "profile");
  if (simt) {
    if (simt->getString("warp_size_measurement")) {
      profile.simtWarpSize =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simt, "warp_size", "warp_size_measurement", "lane", microbench,
              reader, "simt")));
    } else {
      profile.simtWarpSize = reader.integer(*simt, "warp_size", "simt");
    }
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      profile.simtSetupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles",
          &profile.simtSetupConfidence);
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
        profile.simtOps[op] =
            resolveOpProfile(*ops, op, "throughput_scalar_ops_per_system_cycle",
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
      if (const auto *rates =
              reader.object(*camodel, "warp_instructions_per_system_cycle",
                            "simt.camodel_effective"))
        profile.simtPredicateRate =
            reader.number(*rates, "predicate", "simt.camodel_effective.rates");
    }
    if (const auto *shuffle = reader.object(*simt, "shuffle", "simt")) {
      std::string measuredConfidence;
      profile.simtShuffleRate = resolveNumberOrMeasurement(
          *shuffle, "warp_instructions_per_system_cycle",
          "throughput_measurement", "warp_instruction/system_cycle", microbench,
          reader, "simt.shuffle", &measuredConfidence);
      profile.simtShuffleConfidence =
          reader.optionalString(*shuffle, "confidence", measuredConfidence);
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
    if (const auto *resources =
            reader.object(*simt, "stage_resources", "simt")) {
      profile.simtStageResources.scalarOperationsPerCycle =
          reader.number(*resources, "scalar_operations_per_system_cycle",
                        "simt.stage_resources");
      profile.simtStageResources.issueOperationsPerCycle =
          reader.number(*resources, "issue_instructions_per_system_cycle",
                        "simt.stage_resources");
      profile.simtStageResources.spillTransactionsPerCycle =
          reader.number(*resources, "spill_transactions_per_system_cycle",
                        "simt.stage_resources");
      if (const auto *indirect = reader.object(*resources, "indirect_memory",
                                               "simt.stage_resources")) {
        profile.simtStageResources.indirectLoadTransactionsPerCycle =
            reader.number(*indirect, "load_transactions_per_system_cycle",
                          "simt.stage_resources.indirect_memory");
        profile.simtStageResources.indirectStoreTransactionsPerCycle =
            reader.number(*indirect, "store_transactions_per_system_cycle",
                          "simt.stage_resources.indirect_memory");
        profile.simtStageResources.indirectDependencyLatencyCycles =
            reader.number(*indirect, "dependency_latency_system_cycles",
                          "simt.stage_resources.indirect_memory");
      }
      if (const auto *control = reader.object(*resources, "control_flow",
                                              "simt.stage_resources")) {
        profile.simtStageResources.controlFlow.loopBackedgeCycles =
            reader.number(*control, "loop_backedge_system_cycles",
                          "simt.stage_resources.control_flow");
        profile.simtStageResources.controlFlow.conditionalBranchCycles =
            reader.number(*control, "conditional_branch_system_cycles",
                          "simt.stage_resources.control_flow");
        profile.simtStageResources.controlFlow.divergentBranchPenaltyCycles =
            reader.number(*control, "divergent_branch_penalty_system_cycles",
                          "simt.stage_resources.control_flow");
        profile.simtStageResources.controlFlow.synchronizationCycles =
            reader.number(*control, "synchronization_system_cycles",
                          "simt.stage_resources.control_flow");
      }
      if (const auto *superblock =
              reader.object(*resources, "superblock", "simt.stage_resources")) {
        profile.superblockUsefulFactorLimit =
            reader.integer(*superblock, "useful_factor_limit",
                           "simt.stage_resources.superblock");
        profile.superblockPersistentStatePressureFreeFactor =
            reader.integer(*superblock, "persistent_state_pressure_free_factor",
                           "simt.stage_resources.superblock");
        profile.superblockPersistentStateBytesPerCycle = reader.number(
            *superblock, "persistent_state_bytes_per_system_cycle",
            "simt.stage_resources.superblock");
      }
      if (const auto *handoff = reader.object(*resources, "scope_handoff",
                                              "simt.stage_resources")) {
        profile.scopeHandoffFixedDirectionalCycles =
            reader.number(*handoff, "fixed_directional_system_cycles",
                          "simt.stage_resources.scope_handoff");
        profile.scopeSimdUbLoadBytesPerCycle =
            reader.number(*handoff, "simd_ub_load_bytes_per_system_cycle",
                          "simt.stage_resources.scope_handoff");
        profile.scopeSimdUbStoreBytesPerCycle =
            reader.number(*handoff, "simd_ub_store_bytes_per_system_cycle",
                          "simt.stage_resources.scope_handoff");
        profile.scopeSimtUbLoadBytesPerCycle = resolveNumberOrMeasurement(
            *handoff, "simt_ub_load_bytes_per_system_cycle",
            "simt_ub_load_bandwidth_measurement", "byte/system_cycle",
            microbench, reader, "simt.stage_resources.scope_handoff");
        profile.scopeSimtUbStoreBytesPerCycle = resolveNumberOrMeasurement(
            *handoff, "simt_ub_store_bytes_per_system_cycle",
            "simt_ub_store_bandwidth_measurement", "byte/system_cycle",
            microbench, reader, "simt.stage_resources.scope_handoff");
      }
      if (const auto *setupProxy = reader.object(
              *resources, "scope_setup_proxy", "simt.stage_resources")) {
        for (int64_t warps : {1, 2, 4, 8, 16, 32}) {
          std::string key = std::to_string(warps);
          if (const auto *entry = reader.object(
                  *setupProxy, key, "simt.stage_resources.scope_setup_proxy"))
            profile.scopeSetupProxyCycles[warps] = resolveNumberOrMeasurement(
                *entry, "system_cycles", "measurement", "system_cycle",
                microbench, reader,
                "simt.stage_resources.scope_setup_proxy." + key);
        }
        profile.scopeSetupProxyConfidence =
            reader.optionalString(*setupProxy, "confidence", "none");
        profile.scopeSetupProxySource = reader.optionalString(
            *setupProxy, "source", "standalone_empty_vf_proxy");
      }
    }
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (profile.profileVersion != "david-v100-stage-model-20260826-v19")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "unsupported StageModel profile version '%s' "
        "(expected david-v100-stage-model-20260826-v19)",
        profile.profileVersion.c_str());
  const bool usesSharedMicrobench = true;
  if (usesSharedMicrobench && !microbench)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StageModel v19 profile must reference "
                                   "microbenchmark_profile");
  if (*selectionSchemaVersion != 12)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StageModel v19 requires schema_version 12");
  if (profile.simdVectorWidthBits <= 0 || profile.simtWarpSize <= 0 ||
      profile.simdMte2BytesPerCycle <= 0.0 ||
      profile.simdMte3BytesPerCycle <= 0.0 || profile.simtLoadWarpRate <= 0.0 ||
      profile.simtStoreWarpRate <= 0.0 || profile.simtShuffleRate <= 0.0 ||
      profile.simtPredicateRate <= 0.0 ||
      profile.superblockUsefulFactorLimit <= 0 ||
      profile.superblockPersistentStatePressureFreeFactor <= 0 ||
      profile.superblockPersistentStatePressureFreeFactor >
          profile.superblockUsefulFactorLimit ||
      profile.superblockPersistentStateBytesPerCycle <= 0.0 ||
      profile.scopeHandoffFixedDirectionalCycles < 0.0 ||
      profile.scopeSimdUbLoadBytesPerCycle <= 0.0 ||
      profile.scopeSimdUbStoreBytesPerCycle <= 0.0 ||
      profile.scopeSimtUbLoadBytesPerCycle <= 0.0 ||
      profile.scopeSimtUbStoreBytesPerCycle <= 0.0 ||
      profile.scopeSetupProxyCycles.size() != 6 ||
      profile.scopeSetupProxyConfidence != "low" ||
      !profile.simdStageResources.isValid() ||
      !profile.simtStageResources.isValid())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains invalid StageModel rates");
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
        canonicalProfile +
        "\nshared_microbenchmark_sha256=" + profile.microbenchmarkContentSha256;
    llvm::ArrayRef<uint8_t> combinedBytes(
        reinterpret_cast<const uint8_t *>(combinedAssets.data()),
        combinedAssets.size());
    auto combinedHash = llvm::SHA256::hash(combinedBytes);
    profile.contentSha256 =
        llvm::toHex(llvm::ArrayRef<uint8_t>(combinedHash), true);
  }
  return profile;
}

static HardwareProfile
buildStageHardwareProfile(const CandidateProfile &profile, unsigned numWarps) {
  HardwareProfile hardware;
  hardware.profileVersion = profile.profileVersion;
  hardware.target = profile.target;
  hardware.logicalWarpGroupCount = std::max<int64_t>(1, numWarps);
  hardware.superblockUsefulFactorLimit = profile.superblockUsefulFactorLimit;
  hardware.superblockPersistentStatePressureFreeFactor =
      profile.superblockPersistentStatePressureFreeFactor;
  hardware.superblockPersistentStateBytesPerCycle =
      profile.superblockPersistentStateBytesPerCycle;
  hardware.simd.setupCycles = profile.simdSetupCycles;
  hardware.simd.vectorWidth =
      std::max<int64_t>(1, profile.simdVectorWidthBits / 32);
  hardware.simd.issueWidth = hardware.simd.vectorWidth;
  hardware.simt.setupCycles = profile.simtSetupCycles;
  hardware.simt.vectorWidth = 1;
  hardware.simt.issueWidth = std::max<int64_t>(1, profile.simtWarpSize);
  for (const auto &entry : profile.simdOps)
    hardware.simd.operationRates[entry.first()] = {entry.second.throughput,
                                                   entry.second.factor};
  for (const auto &entry : profile.simtOps)
    hardware.simt.operationRates[entry.first()] = {entry.second.throughput,
                                                   entry.second.factor};
  hardware.simd.loadBytesPerCycle = profile.simdMte2BytesPerCycle;
  hardware.simd.storeBytesPerCycle = profile.simdMte3BytesPerCycle;
  hardware.simt.loadWarpInstructionsPerCycle = profile.simtLoadWarpRate;
  hardware.simt.storeWarpInstructionsPerCycle = profile.simtStoreWarpRate;
  const OpProfile simdPredicate = profile.simdOps.lookup("predicate.cmp");
  const OpProfile simtPredicate = profile.simtOps.lookup("predicate.cmp");
  hardware.simd.predicateOperationsPerCycle =
      simdPredicate.throughput / std::max(1.0, simdPredicate.factor);
  hardware.simt.predicateOperationsPerCycle =
      simtPredicate.throughput / std::max(1.0, simtPredicate.factor);
  hardware.simd.shuffleLanesPerCycle = hardware.simd.vectorWidth;
  hardware.simt.shuffleLanesPerCycle =
      profile.simtWarpSize * profile.simtShuffleRate;
  hardware.simd.dotSetupCycles = profile.simdDotSetupCycles;
  hardware.simd.dotFlopsPerCycle = profile.simdDotFlopsPerCycle;
  hardware.simt.dotSetupCycles = profile.simtDotSetupCycles;
  hardware.simt.dotFlopsPerCycle = profile.simtDotFlopsPerCycle;
  hardware.simd.scalarOperationsPerCycle =
      profile.simdStageResources.scalarOperationsPerCycle;
  hardware.simt.scalarOperationsPerCycle =
      profile.simtStageResources.scalarOperationsPerCycle;
  hardware.simd.issueOperationsPerCycle =
      profile.simdStageResources.issueOperationsPerCycle;
  hardware.simt.issueOperationsPerCycle =
      profile.simtStageResources.issueOperationsPerCycle;
  hardware.simd.spillTransactionsPerCycle =
      profile.simdStageResources.spillTransactionsPerCycle;
  hardware.simt.spillTransactionsPerCycle =
      profile.simtStageResources.spillTransactionsPerCycle;
  hardware.simd.indirectLoadTransactionsPerCycle =
      profile.simdStageResources.indirectLoadTransactionsPerCycle;
  hardware.simd.indirectStoreTransactionsPerCycle =
      profile.simdStageResources.indirectStoreTransactionsPerCycle;
  hardware.simd.indirectDependencyLatencyCycles =
      profile.simdStageResources.indirectDependencyLatencyCycles;
  hardware.simt.indirectLoadTransactionsPerCycle =
      profile.simtStageResources.indirectLoadTransactionsPerCycle;
  hardware.simt.indirectStoreTransactionsPerCycle =
      profile.simtStageResources.indirectStoreTransactionsPerCycle;
  hardware.simt.indirectDependencyLatencyCycles =
      profile.simtStageResources.indirectDependencyLatencyCycles;
  hardware.simd.controlFlow = profile.simdStageResources.controlFlow;
  hardware.simt.controlFlow = profile.simtStageResources.controlFlow;
  hardware.transition.simdToSimtCycles =
      profile.scopeHandoffFixedDirectionalCycles;
  hardware.transition.simtToSimdCycles =
      profile.scopeHandoffFixedDirectionalCycles;
  hardware.transition.simdUbLoadBytesPerCycle =
      profile.scopeSimdUbLoadBytesPerCycle;
  hardware.transition.simdUbStoreBytesPerCycle =
      profile.scopeSimdUbStoreBytesPerCycle;
  hardware.transition.simtUbLoadBytesPerCycle =
      profile.scopeSimtUbLoadBytesPerCycle;
  hardware.transition.simtUbStoreBytesPerCycle =
      profile.scopeSimtUbStoreBytesPerCycle;
  hardware.transition.simtWarpSize = profile.simtWarpSize;
  auto setupProxy = profile.scopeSetupProxyCycles.find(numWarps);
  if (setupProxy == profile.scopeSetupProxyCycles.end())
    setupProxy = profile.scopeSetupProxyCycles.find(32);
  hardware.transition.scopeSetupProxyCycles = setupProxy->second;
  hardware.transition.scopeSetupProxyConfidence =
      profile.scopeSetupProxyConfidence;
  hardware.transition.scopeSetupProxySource = profile.scopeSetupProxySource;
  hardware.transition.source =
      "exact scope tensor bytes crossing SIMD/SIMT register files through UB; "
      "directional setup remains unmeasured; standalone SIMT VF setup is "
      "used only by the conservative shadow route";
  return hardware;
}

static llvm::Expected<StageCostModelSummary>
evaluateStageModel(const SimdSimtFeatureSummary &features,
                   const CandidateProfile &profile, unsigned numWarps,
                   bool wholeKernelSuperblockMaterializable,
                   bool scopeSuperblockMaterializable, ModuleOp module,
                   const SimtAnchorPlan &anchorPlan) {
  int64_t maximumSuperblockFactor =
      (wholeKernelSuperblockMaterializable || features.autoBlockifyV1Applied)
          ? 4
          : 1;
  int64_t launchWarpLimit = 64;
  if (module) {
    module.walk([&](Operation *operation) {
      llvm::StringRef name = operation->getName().getStringRef();
      if (name.contains("barrier") || name.contains("sync"))
        launchWarpLimit = 32;
    });
  }
  while (maximumSuperblockFactor > 1 &&
         maximumSuperblockFactor * static_cast<int64_t>(numWarps) >
             launchWarpLimit)
    maximumSuperblockFactor /= 2;
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StageModel requires a non-null ModuleOp");

  HardwareProfile hardwareProfile =
      buildStageHardwareProfile(profile, numWarps);
  ProfileProvider provider(std::move(hardwareProfile));
  auto snapshot = provider.getSnapshot(profile.target, profile.profileVersion);
  if (!snapshot)
    return snapshot.takeError();
  StageDiscoveryOptions discoveryOptions;
  discoveryOptions.tinyDotFlopsMax = profile.structural.tinyDotFlopsMax;
  discoveryOptions.maximumSuperblockFactor = maximumSuperblockFactor;
  auto graph = StageDiscovery().discover(module, anchorPlan, discoveryOptions);
  if (!graph)
    return graph.takeError();
  auto costTable = StageCostEvaluator().evaluate(*graph, **snapshot);
  if (!costTable)
    return costTable.takeError();
  auto routes = solveStageRoutes(*costTable, (*snapshot)->transition);
  if (!routes)
    return routes.takeError();
  return std::move(*routes);
}

static SimtApplicabilityResult
evaluateSimtApplicability(const SimdSimtFeatureSummary &features,
                          bool targetSupported) {
  SimtApplicabilityResult result;
  result.targetSupported = targetSupported;
  result.recognizedAnchorCount = features.simtAnchors.recognizedCount;
  result.materializableAnchorCount = features.simtAnchors.count;
  result.mechanisms = features.simtAnchors.mechanismKinds;
  for (const std::string &kind : features.observedMixedKinds)
    appendUnique(result.mechanisms, kind);
  llvm::sort(result.mechanisms);
  result.mechanismDetected =
      result.recognizedAnchorCount > 0 || !result.mechanisms.empty();
  result.materializable =
      targetSupported && result.materializableAnchorCount > 0;
  if (!result.mechanismDetected)
    result.reasons.push_back("no_recognized_simt_mechanism");
  else if (!targetSupported)
    result.reasons.push_back("target_does_not_support_simt_materialization");
  else if (result.materializableAnchorCount == 0)
    result.reasons.push_back("no_materializable_simt_anchor");
  return result;
}

static llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>>
legalCandidates(const SimdSimtCandidateScores &scores, bool allSimdLegal,
                bool allSimtLegal, bool mixedLegal) {
  llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>> candidates;
  if (allSimdLegal)
    candidates.push_back({scores.allSimd, SimdSimtCandidateKind::AllSIMD});
  if (allSimtLegal)
    candidates.push_back(
        {scores.allSimtOnly, SimdSimtCandidateKind::AllSIMTOnly});
  if (mixedLegal)
    candidates.push_back(
        {scores.mixedSimdSimt, SimdSimtCandidateKind::MixedSIMDSIMT});
  llvm::stable_sort(candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  return candidates;
}

static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores,
                                        bool allSimdLegal, bool allSimtLegal,
                                        bool mixedLegal) {
  return legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal)
      .front()
      .second;
}

static bool sameRoute(const StageRoutePlan &lhs, const StageRoutePlan &rhs) {
  if (lhs.stageIndices != rhs.stageIndices ||
      lhs.implementations.size() != rhs.implementations.size())
    return false;
  for (auto [left, right] :
       llvm::zip_equal(lhs.implementations, rhs.implementations))
    if (left.mode != right.mode ||
        left.superblockFactor != right.superblockFactor)
      return false;
  return true;
}

static SimdSimtCandidateKind
chooseRunnerUp(const SimdSimtCandidateScores &scores, bool allSimdLegal,
               bool allSimtLegal, bool mixedLegal, SimdSimtCandidateKind best) {
  auto candidates =
      legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal);
  return candidates.size() > 1 ? candidates[1].second : best;
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

double SimdSimtCandidateScores::get(SimdSimtCandidateKind candidate) const {
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

static llvm::json::Array toReasonJSON(const std::vector<std::string> &reasons) {
  llvm::json::Array result;
  for (const std::string &reason : reasons)
    result.push_back(reason);
  return result;
}

static llvm::json::Object
toLowerabilityJSON(const CandidateLowerability &lowerability) {
  llvm::json::Object result;
  auto route = [](CandidateLoweringStatus status,
                  const std::vector<std::string> &reasons) {
    llvm::json::Object entry;
    entry["status"] = stringifyCandidateLoweringStatus(status).str();
    entry["reasons"] = toReasonJSON(reasons);
    return entry;
  };
  result[kAllSimd] = route(lowerability.allSimd, lowerability.allSimdReasons);
  result[kAllSimtOnly] =
      route(lowerability.allSimtOnly, lowerability.allSimtOnlyReasons);
  result[kMixedSimdSimt] = route(lowerability.mixed, lowerability.mixedReasons);
  return result;
}

static llvm::json::Object toAtomicFactsJSON(const TensorAtomicFacts &facts) {
  llvm::json::Object result;
  result["update_elements"] = facts.updateElements;
  result["address_rank"] = facts.addressRank;
  result["value_type"] = facts.valueType;
  result["offset_type"] = facts.offsetType;
  result["operation"] = facts.operation;
  result["has_mask"] = facts.hasMask;
  if (facts.staticMaskActiveFraction)
    result["static_mask_active_fraction"] = *facts.staticMaskActiveFraction;
  else
    result["static_mask_active_fraction"] = nullptr;
  result["result_used"] = facts.resultUsed;
  result["address_is_lane_varying"] = facts.addressIsLaneVarying;
  result["address_depends_on_loaded_index"] = facts.addressDependsOnLoadedIndex;
  result["contention"] = facts.contention;
  return result;
}

static llvm::json::Object toHistogramFactsJSON(const HistogramFacts &facts) {
  llvm::json::Object result;
  result["input_elements"] = facts.inputElements;
  result["num_bins"] = facts.numBins;
  result["input_type"] = facts.inputType;
  result["result_type"] = facts.resultType;
  return result;
}

static llvm::json::Object
toPlainCumsumFactsJSON(const PlainCumsumFacts &facts) {
  llvm::json::Object result;
  result["axis_extent"] = facts.axisExtent;
  result["element_type"] = facts.elementType;
  result["reverse"] = facts.reverse;
  return result;
}

static llvm::json::Object
toTriangularSolveFactsJSON(const TriangularSolveFacts &facts) {
  llvm::json::Object result;
  result["block_rows"] = facts.blockRows;
  result["block_columns"] = facts.blockColumns;
  result["accumulator_type"] = facts.accumulatorType;
  result["recurrence_start_row"] = facts.recurrenceStartRow;
  result["recurrence_loop_count"] = facts.recurrenceLoopCount;
  result["dense_dot_tail_ops"] = facts.denseDotTailOps;
  result["requires_cube_tail_partition"] = facts.requiresCubeTailPartition;
  return result;
}

llvm::json::Object SimtAnchorFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["recognized_count"] = recognizedCount;
  result["count"] = count;
  result["materializable_count"] = count;
  result["covered_operation_count"] = coveredOperationCount;
  result["load_ops"] = loadOps;
  result["store_ops"] = storeOps;
  result["reduce_ops"] = reduceOps;
  result["scan_ops"] = scanOps;
  result["gather_ops"] = gatherOps;
  result["dot_ops"] = dotOps;
  result["atomic_ops"] = atomicOps;
  result["histogram_ops"] = histogramOps;
  result["max_tensor_numel"] = maxTensorNumel;
  result["max_element_bits"] = maxElementBits;
  result["mask_rank_sum"] = maskRankSum;
  result["unique_mask_values"] = uniqueMaskValues;
  result["unique_mask_rank_sum"] = uniqueMaskRankSum;
  result["predicate_elements"] = predicateElements;
  result["predicate_lane_evaluations"] = predicateLaneEvaluations;
  result["pointer_tensor_ops"] = pointerTensorOps;
  result["loaded_index_dependent_memory_ops"] = loadedIndexDependentMemoryOps;
  result["lane_dependent_pointer_ops"] = laneDependentPointerOps;
  result["max_reduce_axis_extent"] = maxReduceAxisExtent;
  result["weighted_reduce_axis_elements"] = weightedReduceAxisElements;
  result["shuffle_lane_steps"] = shuffleLaneSteps;
  result["static_loop_count"] = staticLoopCount;
  result["static_loop_trip_count_sum"] = staticLoopTripCountSum;
  result["modeled_dynamic_loop_count"] = modeledDynamicLoopCount;
  result["modeled_dynamic_loop_trip_count_sum"] =
      modeledDynamicLoopTripCountSum;
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["active_lane_ratio"] = activeLaneRatio;
  result["has_control_flow"] = hasControlFlow;
  result["weighted_ops"] = ::toJSON(weightedOps);
  result["op_elements"] = ::toJSON(opElements);
  result["load_bytes"] = loadBytes;
  result["store_bytes"] = storeBytes;
  result["load_warp_instructions"] = loadWarpInstructions;
  result["store_warp_instructions"] = storeWarpInstructions;
  result["dot_flops"] = dotFlops;
  result["captured_tensor_count"] = capturedTensorCount;
  result["escaping_tensor_count"] = escapingTensorCount;
  result["captured_tensor_bytes"] = capturedTensorBytes;
  result["escaping_tensor_bytes"] = escapingTensorBytes;
  llvm::json::Array mechanisms;
  for (const std::string &kind : mechanismKinds)
    mechanisms.push_back(kind);
  result["mechanism_kinds"] = std::move(mechanisms);
  llvm::json::Array atomicFacts;
  for (const TensorAtomicFacts &facts : tensorAtomics)
    atomicFacts.push_back(toAtomicFactsJSON(facts));
  result["tensor_atomics"] = std::move(atomicFacts);
  llvm::json::Array histogramFacts;
  for (const HistogramFacts &facts : histograms)
    histogramFacts.push_back(toHistogramFactsJSON(facts));
  result["histograms"] = std::move(histogramFacts);
  llvm::json::Array cumsumFacts;
  for (const PlainCumsumFacts &facts : plainCumsums)
    cumsumFacts.push_back(toPlainCumsumFactsJSON(facts));
  result["plain_cumsums"] = std::move(cumsumFacts);
  llvm::json::Array triangularFacts;
  for (const TriangularSolveFacts &facts : triangularSolves)
    triangularFacts.push_back(toTriangularSolveFactsJSON(facts));
  result["triangular_solves"] = std::move(triangularFacts);
  result["kernel_lowerability"] = toLowerabilityJSON(kernelLowerability);
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
  result["unique_mask_values"] = uniqueMaskValues;
  result["unique_mask_rank_sum"] = uniqueMaskRankSum;
  result["predicate_elements"] = predicateElements;
  result["predicate_lane_evaluations"] = predicateLaneEvaluations;
  result["mask_broadcast_ops"] = maskBroadcastOps;
  result["pointer_tensor_ops"] = pointerTensorOps;
  result["pointer_unstructured_dims"] = pointerUnstructuredDims;
  result["loaded_index_dependent_memory_ops"] = loadedIndexDependentMemoryOps;
  result["lane_dependent_pointer_ops"] = laneDependentPointerOps;
  result["row_local_reduce_ops"] = rowLocalReduceOps;
  result["max_reduce_axis_extent"] = maxReduceAxisExtent;
  result["weighted_reduce_axis_elements"] = weightedReduceAxisElements;
  result["shuffle_lane_steps"] = shuffleLaneSteps;
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
    dotShapes.push_back(llvm::json::Array({shape[0], shape[1], shape[2]}));
  result["dot_mnk"] = std::move(dotShapes);
  result["static_loop_count"] = staticLoopCount;
  result["static_loop_trip_count_sum"] = staticLoopTripCountSum;
  result["static_loop_trip_count_max"] = staticLoopTripCountMax;
  result["modeled_dynamic_loop_count"] = modeledDynamicLoopCount;
  result["modeled_dynamic_loop_trip_count_sum"] =
      modeledDynamicLoopTripCountSum;
  result["loop_carried_data_dependency_count"] = loopCarriedDataDependencyCount;
  result["pointer_induction_dependency_count"] =
      pointerInductionDependencyCount;
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["active_lane_ratio"] = activeLaneRatio;
  llvm::json::Object postTransform;
  postTransform["ttir_layout_merge_applied"] = ttirLayoutMergeApplied;
  postTransform["coalesce_factor"] = coalesceFactor;
  postTransform["coalesce_axis"] = coalesceAxis;
  postTransform["auto_blockify_v1_applied"] = autoBlockifyV1Applied;
  postTransform["auto_blockify_v1_loop_count"] = autoBlockifyV1LoopCount;
  postTransform["auto_blockify_v1_schedule_op_count"] =
      autoBlockifyV1ScheduleOpCount;
  postTransform["auto_blockify_v1_dynamic_trip_count"] =
      autoBlockifyV1HasDynamicTripCount;
  result["post_transform"] = std::move(postTransform);
  result["has_dot"] = hasDot;
  result["has_gather"] = hasGather;
  result["has_atomic"] = hasAtomic;
  result["has_histogram"] = hasHistogram;
  result["has_scan"] = hasScan;
  result["has_explicit_scope"] = hasExplicitScope;
  result["has_control_flow"] = hasControlFlow;
  result["has_dynamic_shape"] = hasDynamicShape;
  result["has_unknown_trip_count"] = hasUnknownTripCount;
  result["simt_anchors"] = simtAnchors.toJSON();
  llvm::json::Array mixedKinds;
  for (const std::string &kind : observedMixedKinds)
    mixedKinds.push_back(kind);
  result["observed_mixed_kinds"] = std::move(mixedKinds);
  result["mixed_required"] = !observedMixedKinds.empty();
  result["mandatory_mixed_enabled"] = false;
  return result;
}

llvm::json::Object SimtApplicabilityResult::toJSON() const {
  llvm::json::Object result;
  result["mechanism_detected"] = mechanismDetected;
  result["target_supported"] = targetSupported;
  result["materializable"] = materializable;
  result["recognized_anchor_count"] = recognizedAnchorCount;
  result["materializable_anchor_count"] = materializableAnchorCount;
  llvm::json::Array mechanismValues;
  for (const std::string &mechanism : mechanisms)
    mechanismValues.push_back(mechanism);
  result["mechanisms"] = std::move(mechanismValues);
  llvm::json::Array reasonValues;
  for (const std::string &reason : reasons)
    reasonValues.push_back(reason);
  result["reasons"] = std::move(reasonValues);
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
  result["selection_profile_content_sha256"] = selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = microbenchmarkProfileVersion;
  sharedEvidence["target"] = microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] = microbenchmarkProfileContentSha256;
  result["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  result["unit"] = scoreUnit;
  result["score_scope"] = scoreScope;
  result["excludes"] = llvm::json::Array({"host_launch", "grid_wave_count"});
  result["candidate_costs"] = candidateCosts.toJSON();
  result["conservative_candidate_costs"] = conservativeCandidateCosts.toJSON();
  result["candidate_ratios_to_best"] = candidateRatiosToBest.toJSON();
  result["nominal_decision_kind"] = stringifySimdSimtCandidate(nominalDecision);
  result["conservative_decision_kind"] =
      stringifySimdSimtCandidate(conservativeDecision);
  result["transition_sensitive"] = transitionSensitive;
  result["decision_kind"] = stringifySimdSimtCandidate(decision);
  result["best_score"] = bestScore;
  llvm::json::Array selectableCandidates;
  if (allSimdCandidateLegal)
    selectableCandidates.push_back(kAllSimd);
  if (allSimtOnlyCandidateLegal)
    selectableCandidates.push_back(kAllSimtOnly);
  if (mixedCandidateLegal)
    selectableCandidates.push_back(kMixedSimdSimt);
  result["selectable_candidates"] = std::move(selectableCandidates);
  llvm::json::Array unsupportedValues;
  for (const std::string &value : unsupported)
    unsupportedValues.push_back(value);
  result["unmodeled_cost_terms"] = std::move(unsupportedValues);
  result["applicability"] = applicability.toJSON();
  result["stage_model"] = stageModel.toJSON();

  llvm::json::Object roles;
  roles[kAllSimd] =
      allSimdCandidateLegal ? "selectable_candidate" : "inapplicable";
  roles[kAllSimtOnly] =
      allSimtOnlyCandidateLegal ? "selectable_candidate" : "inapplicable";
  roles[kMixedSimdSimt] =
      mixedCandidateLegal ? "selectable_candidate" : "inapplicable";
  result["candidate_roles"] = std::move(roles);

  if (includeFeaturesInJSON)
    result["features"] = features.toJSON();
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
  if (const char *environment = std::getenv("TRITON_ASCEND_SIMD_SIMT_PROFILE"))
    if (*environment)
      return environment;
#ifdef TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH
  return TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH;
#else
  return {};
#endif
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module, bool compileOn91095) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
  return analyzeSimdSimtFeatures(module, anchorPlan);
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module,
                                      const SimtAnchorPlan &anchorPlan) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");

  SimdSimtFeatureSummary features;
  initializeWorkMaps(features);
  initializeWorkMaps(features.simtAnchors);
  features.ttirLayoutMergeApplied =
      module->hasAttr("ta.ttir_layout_merge.applied");
  if (auto factor = module->getAttrOfType<IntegerAttr>("hacc.coalesce_factor"))
    features.coalesceFactor = std::max<int64_t>(1, factor.getInt());
  if (auto axis = module->getAttrOfType<IntegerAttr>("hacc.coalesce_axis"))
    features.coalesceAxis = axis.getInt();
  module.walk([&](Operation *op) {
    if (op->hasAttr("ta.auto_blockify_v1"))
      features.autoBlockifyV1Applied = true;
    if (op->hasAttr("ta.auto_blockify_v1.schedule"))
      ++features.autoBlockifyV1ScheduleOpCount;
    if (!op->hasAttr("ta.auto_blockify_v1.loop"))
      return;
    features.autoBlockifyV1Applied = true;
    ++features.autoBlockifyV1LoopCount;
    if (!getKnownStaticLoopTripCount(op))
      features.autoBlockifyV1HasDynamicTripCount = true;
  });
  llvm::DenseSet<Operation *> anchorSet;
  llvm::DenseMap<Operation *, int64_t> structuralTripEstimates;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable)
      continue;
    if (anchor.scopeOperations.empty()) {
      if (anchor.operation)
        anchorSet.insert(anchor.operation);
    } else {
      for (Operation *scopeOperation : anchor.scopeOperations)
        if (scopeOperation)
          anchorSet.insert(scopeOperation);
    }
    if (anchor.kind == SimtAnchorKind::TriangularSolveLoop) {
      // A recognized solve_tril block has a fixed 16x16 state and starts its
      // recurrence at row 2: a full block performs 16 - 2 = 14 iterations.
      // The TTIR upper bound is min(runtime_remaining, block_end), so it is
      // not a compile-time constant even though the full-tile estimate is
      // structurally known.  Do not apply this fallback to generic loops.
      for (Operation *scopeOperation : anchor.scopeOperations)
        if (scopeOperation &&
            scopeOperation->getName().getStringRef() == "scf.for")
          structuralTripEstimates[scopeOperation] = 14;
    }
  }
  features.simtAnchors.recognizedCount = anchorPlan.anchors.size();
  features.simtAnchors.count = anchorPlan.materializableCount();
  features.simtAnchors.kernelLowerability = anchorPlan.kernelLowerability;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    std::string kind = stringifySimtAnchorKind(anchor.kind).str();
    appendUnique(features.simtAnchors.mechanismKinds, kind);
    appendUnique(features.observedMixedKinds, kind);
    if (const auto *facts = std::get_if<TensorAtomicFacts>(&anchor.facts))
      features.simtAnchors.tensorAtomics.push_back(*facts);
    else if (const auto *facts = std::get_if<HistogramFacts>(&anchor.facts))
      features.simtAnchors.histograms.push_back(*facts);
    else if (const auto *facts = std::get_if<PlainCumsumFacts>(&anchor.facts))
      features.simtAnchors.plainCumsums.push_back(*facts);
    else if (const auto *facts =
                 std::get_if<TriangularSolveFacts>(&anchor.facts))
      features.simtAnchors.triangularSolves.push_back(*facts);
  }

  auto isInAnchor = [&](Operation *op) {
    for (Operation *current = op; current; current = current->getParentOp())
      if (anchorSet.contains(current))
        return true;
    return false;
  };

  llvm::DenseSet<Value> capturedTensors;
  llvm::DenseSet<Value> escapingTensors;
  llvm::DenseSet<Value> uniqueMasks;
  llvm::DenseSet<Value> anchorUniqueMasks;
  auto isValueDefinedInAnchor = [&](Value value) {
    if (Operation *definingOp = value.getDefiningOp())
      return isInAnchor(definingOp);
    auto argument = dyn_cast<BlockArgument>(value);
    Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
    return parent && isInAnchor(parent);
  };
  module.walk([&](Operation *op) {
    if (!isInAnchor(op))
      return;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()) ||
          isValueDefinedInAnchor(operand) ||
          !capturedTensors.insert(operand).second)
        continue;
      ++features.simtAnchors.capturedTensorCount;
      features.simtAnchors.capturedTensorBytes +=
          getStaticTensorBytes(operand.getType());
    }
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || result.use_empty())
        continue;
      bool escapes = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return !isInAnchor(use.getOwner());
      });
      if (!escapes || !escapingTensors.insert(result).second)
        continue;
      ++features.simtAnchors.escapingTensorCount;
      features.simtAnchors.escapingTensorBytes +=
          getStaticTensorBytes(result.getType());
    }
  });
  auto updateTypeStats = [&](Type type, bool inAnchor) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (!tensor.hasStaticShape())
        features.hasDynamicShape = true;
      features.maxElementBits =
          std::max(features.maxElementBits, getTypeBitWidth(type));
      features.maxTensorRank =
          std::max<int64_t>(features.maxTensorRank, tensor.getRank());
      features.maxTensorNumel =
          std::max(features.maxTensorNumel, getStaticNumElements(type));
      if (inAnchor) {
        features.simtAnchors.maxElementBits = std::max(
            features.simtAnchors.maxElementBits, getTypeBitWidth(type));
        features.simtAnchors.maxTensorNumel = std::max(
            features.simtAnchors.maxTensorNumel, getStaticNumElements(type));
      }
    }
  };

  module.walk([&](Operation *op) {
    // V1 scheduling is a separate dispatch phase.  Keep it in post_transform
    // diagnostics, but do not reinterpret its dynamic physical-core loop as
    // an unknown-trip algorithm loop or charge its scalar prologue at
    // candidate-specific SIMD/SIMT rates.
    if (op->hasAttr("ta.auto_blockify_v1.schedule"))
      return;
    llvm::StringRef name = op->getName().getStringRef();
    const int64_t elements = getOperationElements(op);
    const int64_t loopMultiplier =
        getLoopMultiplier(op, structuralTripEstimates);
    const bool inAnchor = isInAnchor(op);
    if (inAnchor)
      ++features.simtAnchors.coveredOperationCount;

    for (Type type : op->getOperandTypes())
      updateTypeStats(type, inAnchor);
    for (Type type : op->getResultTypes())
      updateTypeStats(type, inAnchor);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          updateTypeStats(argument.getType(), inAnchor);

    if (name.starts_with("arith."))
      ++features.arithOps;
    if (name.starts_with("math."))
      ++features.mathOps;
    if (name.starts_with("scf.") || name.starts_with("cf.")) {
      features.hasControlFlow = true;
      if (inAnchor)
        features.simtAnchors.hasControlFlow = true;
    }
    if (name == "scf.if" || name == "cf.cond_br") {
      ++features.conditionalBranchCount;
      if (inAnchor)
        ++features.simtAnchors.conditionalBranchCount;
      const bool laneVarying =
          op->getNumOperands() > 0 &&
          isa<RankedTensorType>(op->getOperand(0).getType());
      if (laneVarying) {
        ++features.divergentBranchCount;
        if (inAnchor)
          ++features.simtAnchors.divergentBranchCount;
      }
    }
    if (name == "scope.scope")
      features.hasExplicitScope = true;

    auto incrementRaw = [&](int64_t &counter, int64_t &anchorCounter) {
      ++counter;
      if (inAnchor)
        ++anchorCounter;
    };
    if (name == "tt.load")
      incrementRaw(features.loadOps, features.simtAnchors.loadOps);
    else if (name == "tt.store")
      incrementRaw(features.storeOps, features.simtAnchors.storeOps);
    else if (name == "tt.reduce")
      incrementRaw(features.reduceOps, features.simtAnchors.reduceOps);
    else if (name == "tt.scan" || name == "tt.associative_scan")
      incrementRaw(features.scanOps, features.simtAnchors.scanOps);
    else if (name == "tt.gather")
      incrementRaw(features.gatherOps, features.simtAnchors.gatherOps);
    else if (name == "tt.dot")
      incrementRaw(features.dotOps, features.simtAnchors.dotOps);
    else if (name.starts_with("tt.atomic"))
      incrementRaw(features.atomicOps, features.simtAnchors.atomicOps);
    else if (name == "tt.histogram")
      incrementRaw(features.histogramOps, features.simtAnchors.histogramOps);
    else if (name == "tt.broadcast")
      ++features.broadcastOps;
    else if (name == "tt.expand_dims")
      ++features.expandDimsOps;
    else if (name == "tt.splat")
      ++features.splatOps;
    else if (name == "tt.addptr")
      ++features.addPtrOps;

    if (name == "arith.addf" || name == "arith.addi")
      ++features.addOps;
    else if (name == "arith.subf" || name == "arith.subi")
      ++features.subOps;
    else if (name == "arith.mulf" || name == "arith.muli")
      ++features.mulOps;
    else if (name == "arith.divf" || name == "arith.divsi" ||
             name == "arith.divui")
      ++features.divOps;
    else if (name == "arith.maxnumf" || name == "arith.maxf" ||
             name == "arith.maxsi" || name == "arith.maxui")
      ++features.maxOps;
    else if (name == "math.absf" || name == "math.absi")
      ++features.absOps;
    else if (name == "math.exp")
      ++features.expOps;
    else if (name == "math.log")
      ++features.logOps;
    else if (name == "arith.cmpf" || name == "arith.cmpi")
      ++features.cmpOps;
    else if (name == "arith.select")
      ++features.selectOps;
    else if (isCastOp(name))
      ++features.castOps;
    else if (name.starts_with("tt.clamp"))
      ++features.clampOps;

    llvm::StringRef weightedKind = classifyWeightedOp(name);
    if (!weightedKind.empty()) {
      int64_t weightedElements = elements;
      if (name == "tt.histogram" && op->getNumOperands() > 0)
        if (auto input =
                dyn_cast<RankedTensorType>(op->getOperand(0).getType()))
          if (input.hasStaticShape())
            weightedElements = getStaticNumElements(input);
      features.weightedOps[weightedKind] += loopMultiplier;
      features.opElements[weightedKind] += weightedElements * loopMultiplier;
      if (inAnchor) {
        features.simtAnchors.weightedOps[weightedKind] += loopMultiplier;
        features.simtAnchors.opElements[weightedKind] +=
            weightedElements * loopMultiplier;
      }
    }

    if (name == "scf.for") {
      auto knownTripCount = getKnownStaticLoopTripCount(op);
      if (!knownTripCount)
        features.hasUnknownTripCount = true;
      int64_t tripCount = getModeledLoopTripCount(op, structuralTripEstimates);
      const bool usedStructuralEstimate =
          !knownTripCount && structuralTripEstimates.contains(op);
      ++features.staticLoopCount;
      features.staticLoopTripCountSum += tripCount;
      features.staticLoopTripCountMax =
          std::max(features.staticLoopTripCountMax, tripCount);
      if (usedStructuralEstimate) {
        ++features.modeledDynamicLoopCount;
        features.modeledDynamicLoopTripCountSum += tripCount;
      }
      if (inAnchor) {
        ++features.simtAnchors.staticLoopCount;
        features.simtAnchors.staticLoopTripCountSum += tripCount;
        if (usedStructuralEstimate) {
          ++features.simtAnchors.modeledDynamicLoopCount;
          features.simtAnchors.modeledDynamicLoopTripCountSum += tripCount;
        }
      }
      if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
        Block &body = op->getRegion(0).front();
        // scf.for block argument 0 is the induction variable; remaining
        // arguments are loop-carried iter_args.
        for (unsigned argumentIndex = 1; argumentIndex < body.getNumArguments();
             ++argumentIndex) {
          BlockArgument argument = body.getArgument(argumentIndex);
          if (argument.use_empty())
            continue;
          if (isPointerType(argument.getType()) ||
              isAddressOnlyLoopCarriedValue(argument))
            ++features.pointerInductionDependencyCount;
          else
            ++features.loopCarriedDataDependencyCount;
        }
      }
    }

    auto dataTypeAndElements = [&](bool load) -> std::pair<Type, int64_t> {
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
          static_cast<int64_t>(std::ceil(dataElements / 32.0)) * loopMultiplier;
      if (load) {
        features.loadBytes += bytes;
        features.loadWarpInstructions += warpInstructions;
        if (inAnchor) {
          features.simtAnchors.loadBytes += bytes;
          features.simtAnchors.loadWarpInstructions += warpInstructions;
        }
      } else {
        features.storeBytes += bytes;
        features.storeWarpInstructions += warpInstructions;
        if (inAnchor) {
          features.simtAnchors.storeBytes += bytes;
          features.simtAnchors.storeWarpInstructions += warpInstructions;
        }
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
          if (inAnchor)
            features.simtAnchors.dotFlops += 2 * m * n * k * loopMultiplier;
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
      if (op->getNumOperands() > 0) {
        auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        if (source && source.hasStaticShape() && axis) {
          int64_t axisValue = axis.getInt();
          if (axisValue < 0)
            axisValue += source.getRank();
          if (axisValue >= 0 && axisValue < source.getRank()) {
            int64_t extent = source.getShape()[axisValue];
            if (extent > 0) {
              features.maxReduceAxisExtent =
                  std::max(features.maxReduceAxisExtent, extent);
              features.weightedReduceAxisElements += extent * loopMultiplier;
              const int64_t shuffleLevels = static_cast<int64_t>(
                  std::ceil(std::log2(static_cast<double>(extent))));
              const int64_t inputElements = getStaticNumElements(source);
              const int64_t shuffleLaneSteps =
                  inputElements * shuffleLevels * loopMultiplier;
              features.shuffleLaneSteps += shuffleLaneSteps;
              if (inAnchor) {
                features.simtAnchors.maxReduceAxisExtent =
                    std::max(features.simtAnchors.maxReduceAxisExtent, extent);
                features.simtAnchors.weightedReduceAxisElements +=
                    extent * loopMultiplier;
                features.simtAnchors.shuffleLaneSteps += shuffleLaneSteps;
              }
            }
          }
        }
      }
    }

    if ((name == "tt.scan" || name == "tt.associative_scan") &&
        op->getNumOperands() > 0) {
      auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (source && source.hasStaticShape() && axis) {
        int64_t axisValue = axis.getInt();
        if (axisValue < 0)
          axisValue += source.getRank();
        if (axisValue >= 0 && axisValue < source.getRank()) {
          int64_t extent = source.getShape()[axisValue];
          if (extent > 0) {
            const int64_t shuffleLevels = static_cast<int64_t>(
                std::ceil(std::log2(static_cast<double>(extent))));
            const int64_t laneSteps =
                getStaticNumElements(source) * shuffleLevels * loopMultiplier;
            features.shuffleLaneSteps += laneSteps;
            if (inAnchor)
              features.simtAnchors.shuffleLaneSteps += laneSteps;
          }
        }
      }
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
      for (int64_t rank : maskRanks) {
        features.maskRankSum += rank;
        if (inAnchor)
          features.simtAnchors.maskRankSum += rank;
      }
      auto addPredicateLaneEvaluations = [&](Type type) {
        if (!isMaskTensorType(type))
          return;
        const int64_t laneEvaluations =
            getStaticNumElements(type) * loopMultiplier;
        features.predicateLaneEvaluations += laneEvaluations;
        if (inAnchor)
          features.simtAnchors.predicateLaneEvaluations += laneEvaluations;
      };
      for (Type type : op->getOperandTypes())
        addPredicateLaneEvaluations(type);
      for (Type type : op->getResultTypes())
        addPredicateLaneEvaluations(type);
      if (name == "tt.broadcast" || name == "tt.expand_dims")
        ++features.maskBroadcastOps;
    }

    auto recordUniqueMask = [&](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.getElementType().isInteger(1))
        return;
      if (uniqueMasks.insert(value).second) {
        ++features.uniqueMaskValues;
        features.uniqueMaskRankSum += type.getRank();
        const int64_t elements = getStaticNumElements(type);
        features.predicateElements += elements;
      }
      if (inAnchor && anchorUniqueMasks.insert(value).second) {
        ++features.simtAnchors.uniqueMaskValues;
        features.simtAnchors.uniqueMaskRankSum += type.getRank();
        const int64_t elements = getStaticNumElements(type);
        features.simtAnchors.predicateElements += elements;
      }
    };
    for (Value operand : op->getOperands())
      recordUniqueMask(operand);
    for (Value result : op->getResults())
      recordUniqueMask(result);

    bool isPointerOperation =
        name == "tt.addptr" || name == "tt.load" || name == "tt.store";
    if (isLoadedIndexDependentMemoryOp(op)) {
      ++features.loadedIndexDependentMemoryOps;
      if (inAnchor)
        ++features.simtAnchors.loadedIndexDependentMemoryOps;
    }
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
        if (inAnchor)
          ++features.simtAnchors.pointerTensorOps;
        maxPointerRank = std::max(maxPointerRank, rank);
        if (rank > 1)
          features.pointerUnstructuredDims += rank;
      }
      if (maxPointerRank > 1) {
        ++features.laneDependentPointerOps;
        if (inAnchor)
          ++features.simtAnchors.laneDependentPointerOps;
      }
    }

    bool anyRankedType = llvm::any_of(op->getOperandTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });
    anyRankedType |= llvm::any_of(op->getResultTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });
    if (name == "tt.load" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarLoadOps;
    if (name == "tt.store" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarStoreOps;
    if (name == "tt.splat" && op->getNumOperands() > 0 &&
        op->getNumResults() > 0 && isPointerType(op->getOperand(0).getType()) &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      ++features.vectorPtrSplatOps;
  });

  features.scalarOps = features.addOps + features.subOps + features.mulOps +
                       features.divOps + features.maxOps + features.absOps +
                       features.expOps + features.logOps + features.cmpOps +
                       features.selectOps + features.castOps +
                       features.clampOps;
  features.hasDot = features.dotOps > 0;
  features.hasGather = features.gatherOps > 0;
  features.hasAtomic = features.atomicOps > 0;
  features.hasHistogram = features.histogramOps > 0;
  features.hasScan = features.scanOps > 0;
  features.rank1IndirectVectorReduce =
      features.maxTensorRank == 1 && features.reduceOps > 0 &&
      features.vectorReduceToScalarOps > 0 && features.vectorPtrSplatOps > 0 &&
      features.scalarLoadOps >= 2;

  return features;
}

static llvm::Expected<SimdSimtCostReport>
estimateStageCandidatesImpl(const SimdSimtFeatureSummary &features,
                            const SimdSimtCostModelOptions &options,
                            ModuleOp module, const SimtAnchorPlan &anchorPlan) {
  auto profileOrError = loadCandidateProfile(options.profilePath);
  if (!profileOrError)
    return profileOrError.takeError();
  CandidateProfile profile = std::move(*profileOrError);

  SimdSimtCostReport report;
  report.profileVersion = profile.profileVersion;
  report.profileTarget = profile.target;
  report.actualTarget = options.actualTarget;
  report.profileContentSha256 = profile.contentSha256;
  report.selectionProfileContentSha256 = profile.selectionContentSha256;
  report.microbenchmarkProfileVersion = profile.microbenchmarkProfileVersion;
  report.microbenchmarkProfileTarget = profile.microbenchmarkProfileTarget;
  report.microbenchmarkProfileContentSha256 =
      profile.microbenchmarkContentSha256;
  report.scoreUnit = profile.scoreUnit;
  report.targetCompatible = targetMatches(profile, options.actualTarget);
  report.features = features;
  report.applicability =
      evaluateSimtApplicability(features, options.compileOn91095);
  report.includeFeaturesInJSON = options.includeFeaturesInJSON;

  const int64_t numWarps =
      std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
  auto stageModel = evaluateStageModel(
      features, profile, static_cast<unsigned>(numWarps),
      options.wholeKernelSuperblockMaterializable,
      options.scopeSuperblockMaterializable, module, anchorPlan);
  if (!stageModel)
    return stageModel.takeError();
  report.stageModel = std::move(*stageModel);

  report.allSimdCandidateLegal = report.stageModel.allSimd.legal;
  report.allSimtOnlyCandidateLegal = options.compileOn91095 &&
                                     !features.hasExplicitScope &&
                                     report.stageModel.allSimt.legal;
  report.mixedCandidateLegal = options.compileOn91095 &&
                               !features.hasExplicitScope &&
                               report.stageModel.mixed.legal;
  if (report.stageModel.mixed.legal) {
    report.applicability.mechanismDetected = true;
    report.applicability.materializable = options.compileOn91095;
    appendUnique(report.applicability.mechanisms, "generic_stage_seed");
    if (options.compileOn91095)
      report.applicability.reasons.clear();
  }

  report.candidateCosts.allSimd = report.stageModel.allSimd.totalCycles;
  report.candidateCosts.allSimtOnly = report.stageModel.allSimt.totalCycles;
  report.candidateCosts.mixedSimdSimt = report.stageModel.mixed.totalCycles;
  report.conservativeCandidateCosts = report.candidateCosts;
  report.conservativeCandidateCosts.mixedSimdSimt =
      report.stageModel.conservativeMixed.totalCycles;
  const unsigned legalCandidateCount =
      static_cast<unsigned>(report.allSimdCandidateLegal) +
      static_cast<unsigned>(report.allSimtOnlyCandidateLegal) +
      static_cast<unsigned>(report.mixedCandidateLegal);
  if (legalCandidateCount == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "StageModel found no materializable route candidate");
  report.nominalDecision =
      chooseBest(report.candidateCosts, report.allSimdCandidateLegal,
                 report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
  report.conservativeDecision = chooseBest(
      report.conservativeCandidateCosts, report.allSimdCandidateLegal,
      report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
  report.transitionSensitive =
      report.nominalDecision != report.conservativeDecision ||
      (report.nominalDecision == SimdSimtCandidateKind::MixedSIMDSIMT &&
       !sameRoute(report.stageModel.mixed,
                  report.stageModel.conservativeMixed));
  report.decision = report.transitionSensitive ? SimdSimtCandidateKind::AllSIMD
                                               : report.nominalDecision;
  report.bestScore = report.candidateCosts.get(report.decision);
  const double denominator = std::max(1.0e-9, report.bestScore);
  report.candidateRatiosToBest = {
      report.candidateCosts.allSimd / denominator,
      report.candidateCosts.allSimtOnly / denominator,
      report.candidateCosts.mixedSimdSimt / denominator};
  return report;
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimdSimtCostModelOptions &options) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan =
      buildMixedSimtAnchorPlan(module, options.compileOn91095);
  return analyzeSimdSimtCandidates(module, anchorPlan, options);
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimtAnchorPlan &anchorPlan,
    const SimdSimtCostModelOptions &options) {
  auto features = analyzeSimdSimtFeatures(module, anchorPlan);
  if (!features)
    return features.takeError();
  return estimateStageCandidatesImpl(*features, options, module, anchorPlan);
}
