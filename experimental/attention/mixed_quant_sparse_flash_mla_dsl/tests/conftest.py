"""Optional Excel case source for the existing MQSMLA pytest entry."""

import os


def pytest_addoption(parser):
    group = parser.getgroup('mqsmla')
    group.addoption('--excel', dest='mqsmla_excel', default=os.environ.get('MQSMLA_EXCEL'),
                    help='Excel workbook; replaces test_case_paramset when specified')
    group.addoption('--sheet', dest='mqsmla_sheet', default=os.environ.get('MQSMLA_SHEET', 'decode'),
                    help='Excel sheet name (default: decode)')
    group.addoption('--data-mode', choices=('run', 'save', 'replay', 'save-run'), default='run',
                    help='Generate/run, save CPU data, replay .pt, or save then replay each case')
    group.addoption('--pt-dir', default=os.environ.get('MQSMLA_PT_DIR'),
                    help='Directory for saved CPU inputs and golden (.pt)')
    group.addoption('--device-id', type=int, default=0, help='NPU device index (default: 0)')
