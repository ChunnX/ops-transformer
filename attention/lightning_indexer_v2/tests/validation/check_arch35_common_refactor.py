#!/usr/bin/env python3
"""Source-level migration checks; these do not replace AscendC compilation/STC.

Run from any directory with Python 3. Baseline can be overridden with --baseline.
No SDK, device, or third-party Python packages are required.
"""
import argparse
import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
LI = 'attention/lightning_indexer_v2/op_kernel/arch35/'
QLI = 'attention/quant_lightning_indexer_v2/op_kernel/arch35/'
COMMON = LI + 'common/'
BASELINE = 'd3d2ef72493faca6b0c00eaceb94df90049796bf'


def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


def before(path):
    return subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT).decode('utf-8')


def function(source, name):
    pattern = r'(?m)^[^\n]*(?:void|uint32_t|uint64_t)\s+(?:[\w:<>]+::)?' + name + r'\('
    for match in re.finditer(pattern, source):
        start = source.find('{', match.end())
        semi = source.find(';', match.end())
        if start < 0 or 0 <= semi < start:
            continue
        depth = 1
        end = start + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        return source[start + 1:end - 1]
    raise AssertionError(f'Missing definition: {name}')


def tokens(source):
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    source = source.replace('AscendC::', '').replace('Derived::', '').replace('this->', '')
    return re.sub(r'\s+', '', source)


class RefactorChecks(unittest.TestCase):
    def test_kernel_shared_utilities(self):
        shared = read(COMMON + 'lightning_indexer_v2_kernel_base_arch35.h')
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            previous = before(directory + prefix + '_kernel_arch35.h')
            for name in ('GetBN2Idx', 'InitBuffers', 'ProcessDecode'):
                body = function(shared, name)
                body = re.sub(r'auto &kernel = [^;]+;', '', body).replace('kernel.', '')
                self.assertEqual(tokens(function(previous, name)), tokens(body))

    def test_padding_retains_value_dtype_and_events(self):
        shared = read(COMMON + 'lightning_indexer_v2_service_vector_base_arch35.h')
        for directory, prefix, count, flag, bits, infinity in (
                (LI, 'lightning_indexer_v2', 'topk', 'returnValueFlag', 'uint32_t', 'NEG_INF_FLOAT'),
                (QLI, 'quant_lightning_indexer_v2', 'sparseCount', 'returnValue', 'uint16_t', 'NEG_INF_BFLOAT')):
            old = function(before(directory + prefix + '_service_vector_arch35.h'), 'DoTndPadding')
            new = function(shared, 'PadTndOutputs')
            new = re.sub(r'auto &service = [^;]+;', '', new).replace('service.', '')
            new = new.replace('CleanInvalidIndices(paddingOffset, dealSize);',
                              'GlobalTensor<int32_t> indiceOutPaddingStart = indiceOutGm[paddingOffset];'
                              'InitGlobalMemory(indiceOutPaddingStart, dealSize, constInfo_.INVALID_IDX);')
            for key, value in {'topk': 'constInfo_.' + count, 'returnValue': 'constInfo_.' + flag,
                               'ValueBits': bits, 'negativeInfinity': 'constInfo_.' + infinity}.items():
                new = re.sub(r'\b' + key + r'\b', value, new)
            self.assertEqual(tokens(old), tokens(new))

    def test_kernel_loop_and_sync_order(self):
        common = function(read(COMMON + 'lightning_indexer_v2_kernel_base_arch35.h'), 'ProcessMain')
        common = re.sub(r'auto &\w+ = [^;]+;', '', common).replace('kernel.', '')
        common = re.sub(r'ConstInfoT::(?:SYNC_MODE4)', 'SYNC_MODE', common)
        common = common.replace('ConstInfoT::', '')
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            path = directory + prefix + '_kernel_arch35.h'
            old = function(before(path), 'ProcessMain')
            old = re.sub(r'(?:LIV2Common|QLIV2Common)::ConstInfo::(?:LI_SYNC_MODE4|QLIV2_SYNC_MODE4)',
                         'SYNC_MODE', old)
            old = re.sub(r'(?:LIV2Common|QLIV2Common)::ConstInfo::', '', old)
            old = re.sub(r'(?:LIV2Common|QLIV2Common)::RunInfo', 'RunInfoT', old)
            new = common.replace('InitLoopState(runInfo);', '')
            if directory == LI:
                new = new.replace('BeginS2Loop(runInfo);', '').replace('EndS2Loop(runInfo);', '')
            else:
                new = new.replace('uint32_t gloop = 0;', 'uint32_t gloop = 0; uint32_t qScaleLoop = 0;')
                new = new.replace('BeginS2Loop(runInfo);', 'uint32_t kScaleLoop = 0;')
                new = new.replace('EndS2Loop(runInfo);', '++qScaleLoop;')
                adapter = function(read(path), 'ProcessBaseBlock')
                adapter = adapter.replace('loop,', 'gloop,').replace('runInfo.qScaleLoop', 'qScaleLoop')
                adapter = adapter.replace('runInfo.kScaleLoop', 'kScaleLoop')
                new = new.replace('ProcessBaseBlock(gloop, s2LoopIdx, runInfo);', adapter)
                self.assertIn('runInfo.qScaleLoop = 0;', function(read(path), 'InitLoopState'))
                self.assertIn('runInfo.kScaleLoop = 0;', function(read(path), 'BeginS2Loop'))
                self.assertIn('++runInfo.qScaleLoop;', function(read(path), 'EndS2Loop'))
            self.assertEqual(tokens(old), tokens(new))

    def test_metadata_loading_and_end_boundary(self):
        common = function(read(COMMON + 'lightning_indexer_v2_kernel_base_arch35.h'), 'LoadSplitCoreInfo')
        common = re.sub(r'auto &\w+ = [^;]+;', '', common).replace('kernel.', '')
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            path = directory + prefix + '_kernel_arch35.h'
            current = read(path)
            body = common.replace('MetadataTraits::GetIndex', 'GetAttrAbsIndex')
            for key, value in re.findall(r'static constexpr uint32_t (k\w+) = (\w+);', current):
                body = body.replace('MetadataTraits::' + key, value)
            body = body.replace('UpdateLdOutputOffset();', function(current, 'UpdateLdOutputOffset'))
            old = function(before(path), 'SplitCoreByAICPU').replace('ldWorkspaceNumINDEX', 'ldWorkspaceNumIndex')
            self.assertEqual(tokens(old), tokens(body))

    def test_histogram_instruction_order(self):
        shared = function(read(COMMON + 'vf/vf_topk_base_v2.h'), 'StoreHistogramResult')
        for directory, file in ((LI, 'vf_topk_gather_v2.h'), (QLI, 'vf_topk.h')):
            path = directory + 'vf/' + file
            for word in ('First', 'Second', 'Third', 'Last'):
                name = 'Histograms' + word + 'VFImpl'
                old = function(before(path), name)
                new = function(read(path), name)
                def expand(body):
                    body = re.sub(r'(?:liV2TopkCommon::)?StoreHistogramResult\([^;]+\);', lambda _: shared, body)
                    body = re.sub(r'Reg::RegTensor<[^;]+>\s+cout[01]U32(?:Even|Odd);', '', body)
                    body = re.sub(r'static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_\w+\s*=.*?;',
                                  '', body, flags=re.S)
                    return tokens(body)
                self.assertEqual(expand(old), expand(new), path + ':' + name)

    def test_source_and_packaged_includes(self):
        # Model the op_kernel -> <operator>/ installed directory mapping.
        headers = list((ROOT / LI).rglob('*.h')) + list((ROOT / QLI).rglob('*.h'))
        installed = {PurePosixPath(p.relative_to(ROOT).as_posix().replace('/op_kernel/', '/'))
                     for p in headers}
        for path in headers:
            source = path.read_text(encoding='utf-8')
            for include in re.findall(r'^#include "([^"]+)"', source, re.M):
                if 'common/' not in include or 'lightning_indexer_v2' not in include:
                    continue
                if (path.parent / include).is_file():
                    continue
                packed = PurePosixPath(path.relative_to(ROOT).as_posix().replace('/op_kernel/', '/')).parent
                parts = []
                for part in (packed / include).parts:
                    if part == '..':
                        parts.pop()
                    elif part != '.':
                        parts.append(part)
                self.assertIn(PurePosixPath(*parts), installed, f'{path}: {include}')

    def test_common_dependency_direction_and_aliases(self):
        for path in (ROOT / COMMON).rglob('*.h'):
            source = path.read_text(encoding='utf-8')
            self.assertNotRegex(source, r'#include[^\n]*quant_lightning_indexer_v2')
            self.assertNotRegex(source, r'\bQLIV2\w*|\bvirtual\b')
            self.assertNotRegex(source, r'[ \t]+\n', str(path))
        for folder in (LI, QLI):
            for path in (ROOT / folder).rglob('*.h'):
                self.assertNotRegex(path.read_text(encoding='utf-8'),
                                    r'\bstruct\s+(?:LIV2Common|QLIV2Common)::(?:LdSplitCoreInfo|SplitCoreInfo)')

    def test_existing_vector1_base_is_unchanged(self):
        self.assertEqual(before(LI + 'vf/common/lightning_indexer_v2_vector1_base.h'),
                         read(COMMON + 'vf/lightning_indexer_v2_vector1_base.h'))

    def test_cube_event_order(self):
        common = read(COMMON + 'lightning_indexer_v2_service_cube_base_arch35.h')
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            old = before(directory + prefix + '_service_cube_arch35.h')
            for name in ('AllocEventID', 'FreeEventID'):
                self.assertEqual(tokens(function(old, name)), tokens(function(common, name)))

    def test_vector_event_order(self):
        common = read(COMMON + 'lightning_indexer_v2_service_vector_base_arch35.h')
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            path = directory + prefix + '_service_vector_arch35.h'
            for verb in ('Alloc', 'Free'):
                body = function(read(path), verb + 'EventID')
                for suffix in ('VectorOutputEvents', 'TopkEvents'):
                    body = body.replace('this->' + verb + suffix + '();', function(common, verb + suffix))
                self.assertEqual(tokens(function(before(path), verb + 'EventID')), tokens(body))

    def test_mask_formula_preserves_signedness_and_ratio(self):
        common = function(read(COMMON + 'lightning_indexer_v2_kernel_base_arch35.h'), 'GetMaskedS2BaseBlockNum')
        common = re.sub(r'    if \(validLength != nullptr\) \{.*?\n    }', '', common, flags=re.S)
        for field in ('s1BaseSize', 's2BaseSize', 'cmpRatio'):
            common = re.sub(r'\b' + field + r'\b', 'constInfo.' + field, common)
        for directory, prefix in ((LI, 'lightning_indexer_v2'), (QLI, 'quant_lightning_indexer_v2')):
            old = function(before(directory + prefix + '_kernel_arch35.h'), 'GetS2BaseBlockNumOnMask')
            old = old.replace('    tempLoopInfo.validS2Len = validS2Len;', '')
            self.assertEqual(tokens(old), tokens(common))

    def test_target_bin_instruction_order(self):
        common = read(COMMON + 'vf/vf_topk_base_v2.h')
        helper = function(common, 'FindTargetBinAndUpdateNextK')
        for directory, file in ((LI, 'vf_topk_gather_v2.h'), (QLI, 'vf_topk.h')):
            old = before(directory + 'vf/' + file)
            new = read(directory + 'vf/' + file)
            for i, word in enumerate(('First', 'Second', 'Third')):
                name = 'Find' + word + 'TargetBinVFImpl'
                body = function(new, name)
                expanded = helper
                replacements = {'idxBuf': f'idx{i}Buf', 'btmK': 'btmK' + (str(i) if i else '')}
                for key, value in replacements.items():
                    expanded = re.sub(r'\b' + key + r'\b', value, expanded)
                body = re.sub(r'liV2TopkCommon::FindTargetBinAndUpdateNextK\([^;]+\);', lambda _: expanded, body)
                previous = function(old, name)
                if directory == QLI:
                    old_helper = function(old, 'FindTargetBinAndUpdateNextK')
                    for key, value in replacements.items():
                        old_helper = re.sub(r'\b' + key + r'\b', value, old_helper)
                    previous = re.sub(r'FindTargetBinAndUpdateNextK\([^;]+\);', lambda _: old_helper, previous)
                # Compare executable statements; local register declarations can move into the callee.
                def instructions(text):
                    text = re.sub(r'\b(?:sqzIdx|alignIdx|idxPrev|idx|preg)[012]\b',
                                  lambda m: {'preg': 'pregZero'}.get(m[0][:-1], m[0][:-1]), text)
                    text = re.sub(r'Reg::(?:RegTensor<[^;]+>|MaskReg|UnalignRegForStore)\s+\w+;', '', text)
                    return tokens(text)
                self.assertEqual(instructions(previous), instructions(body), name)

    def test_index_offset_vf_is_unchanged(self):
        shared = function(read(COMMON + 'vf/vf_topk_base_v2.h'), 'IndicesAddOffsetVF')
        for path in (LI + 'vf/vf_topk_gather_v2.h', QLI + 'vf/vf_topk_16_gather_quant_v2.h'):
            self.assertEqual(tokens(function(before(path), 'IndicesAddOffsetVF')), tokens(shared))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=BASELINE)
    args, remaining = parser.parse_known_args()
    BASELINE = args.baseline
    unittest.main(argv=[__file__, *remaining])
