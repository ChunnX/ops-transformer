/* Diagnostic-only shim: this protocol has no Matmul/KFC workspace or clients.
 *
 * CANN 9.1 OPC generates matmul::clearWorkspace() for MIX kernels. The real
 * implementation initializes KFC queues and sends Cube->Vector mode2 flag15.
 * That would violate the minimal protocol and contaminate the flag15 control.
 * Supply a no-op exclusively in this standalone experiment translation unit.
 * Never include this shim in kernels using Matmul, KFC, or that workspace.
 */
#ifndef SLIG_CV_VV_CANN_MIX_WRAPPER_SHIM_H
#define SLIG_CV_VV_CANN_MIX_WRAPPER_SHIM_H
namespace matmul {
__aicore__ inline void clearWorkspace(GM_ADDR workspace)
{
    (void)workspace;
}
} // namespace matmul
#endif
