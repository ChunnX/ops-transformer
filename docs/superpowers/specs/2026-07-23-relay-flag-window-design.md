# MegaMoe Layered Relay Flag Window Design

## Goal

Replace per-token polling of flags embedded in relay data records with a
standalone flag window. Both local-relay and remote-relay receive paths load a
complete source-server flag slice into UB, scan all required token IDs without
waiting on an individual token, and stop when every requested token has been
received.

## Scope

The change applies only to the arch35 URMA layered-dispatch path. The MTE
topology, non-layered `MegaMoe` implementation, mask exchange, GMM, and combine
flows remain unchanged.

Supporting changes are required in:

- the arch35 peer-memory layout;
- the layered dispatch sender and local relay producer;
- relay-state reset;
- the local and remote relay receive paths;
- the per-core L2 scratch allocation used for a remote flag snapshot.

## Peer-Memory Layout

The layered URMA peer-memory layout is:

```text
rankSyncInWorldPtr
maskRecvPtr
dispatchRecivePtr     serverNum * bs * relayDataRecordBytes
dispatchFlagPtr       serverNum * bs * ALIGN_32
combineSendPtr
```

`dispatchRecivePtr` contains only quantized token data and scale data. It does
not contain a token ID or ready flag. `relayDataRecordBytes` is the token-scale
payload rounded up to the existing relay-record alignment.

`dispatchFlagPtr` is indexed as `[sourceServer][tokenId]`. Each entry occupies
`ALIGN_32` bytes. The first `uint64_t` of an entry is the ready value:

```text
0 = token data is not ready
1 = token data is ready
```

The remaining bytes are padding that keeps independently updated flags on
separate 32-byte slots.

## Producer Ordering

The first-level dispatch workspace record keeps its existing token ID and
sender-state metadata because the compute cores must communicate the token ID
to the sender cores.

For a locally produced relay record:

1. quantize and write token plus scale to `dispatchRecivePtr`;
2. wait until the payload write is complete;
3. write `uint64_t(1)` to the matching `dispatchFlagPtr` entry.

For a remote relay record:

1. read the token ID from the first-level dispatch workspace metadata;
2. write token plus scale to the remote `dispatchRecivePtr` record;
3. complete or drain the payload write;
4. write or notify `uint64_t(1)` at the remote `dispatchFlagPtr` entry;
5. mark the first-level workspace record as sent.

No producer writes a ready flag into the data record.

## Flag Snapshot and Receive State

The receiver aliases the contiguous storage currently assigned to
`copyTmpTensors_` as:

```text
[bs * ALIGN_32 flag snapshot][copyNum receive-state entries][temporary data area]
```

The exact aliases are established by the existing UB-address-based allocation
style. The implementation must verify that the maximum concurrent aliases fit
inside the reusable temporary region and must not extend a `LocalTensor` past
its assigned UB storage.

Each receive-state entry is zero or one and is indexed by `copyIdx`, not by
token ID:

```text
0 = destination row has not been populated
1 = destination row has been populated
```

The receive-state table is cleared every time a relay receive function is
entered. `receivedCount` is incremented only when an entry changes from zero to
one, so it is always equal to the sum of the table.

## Remote Relay Flow

Each scan round performs:

1. use URMA to read the complete
   `dispatchFlagPtr[srcServer][0..bs)` slice from the remote GM window into the
   calling core's `dispatchL2CommPtr` scratch;
2. drain the URMA read;
3. use MTE2 `DataCopy` to move the complete local-GM scratch snapshot into the
   flag-snapshot `LocalTensor`;
4. wait for GM-to-UB completion before scalar flag reads;
5. iterate all requested `copyIdx` values:
   - skip entries already marked received;
   - obtain `tokenId` from `tripleTensor_`;
   - read the first `uint64_t` in the token's 32-byte flag slot;
   - if the value is zero, continue to the next `copyIdx`;
   - if the value is one, issue token and scale reads to that `copyIdx`'s
     destination and mark it received;
6. repeat from step 1 while `receivedCount < copyNum`;
7. drain outstanding token and scale reads before returning.

The per-core L2 scratch stride is at least
`max(DISPATCH_BUFFER_NUM * sendWorkspaceRecordBytes, bs * ALIGN_32)`, so one
core's full flag snapshot cannot overwrite another core's scratch.

## Local Relay Flow

Each scan round performs:

1. use MTE2 `DataCopy` to move the complete local
   `dispatchFlagPtr[srcServer][0..bs)` slice from GM into the same
   flag-snapshot `LocalTensor`;
2. wait for GM-to-UB completion;
3. scan requested token IDs with the same receive-state rules as the remote
   path;
4. for each newly ready token, move token plus scale through reusable UB
   temporary storage into the corresponding destination row;
5. repeat while `receivedCount < copyNum`.

The local path therefore uses the standalone flag window and no longer waits on
an embedded record flag.

## Reset

`ResetDispatchState` clears all
`serverNum * bs * ALIGN_32 / sizeof(uint64_t)` flag-window elements before the
cross-rank synchronization that starts dispatch. Existing first-level
workspace counters and states are still reset. The receive-state table is
cleared independently at every receive-function entry.

## Verification

Verification covers:

- layout arithmetic proving that data, flag, and combine regions do not
  overlap;
- sender ordering proving that a ready flag cannot become visible before its
  payload;
- full-slice remote GM to local GM to UB synchronization;
- full-slice local GM to UB synchronization;
- out-of-order flags allowing later ready tokens to progress;
- receive-state entries preventing duplicate destination writes;
- termination exactly when `receivedCount == copyNum`;
- zero-token handling without entering the polling loop;
- arch35 MegaMoe kernel compilation and available formatting/static checks.
