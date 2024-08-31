# By Forge


import torch


def native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits(x):
    x = x.view(torch.uint8).view(x.size(0), -1)
    unpacked = torch.stack([x & 15, x >> 4], dim=-1)
    reshaped = unpacked.view(x.size(0), -1)
    reshaped = reshaped.view(torch.int8) - 8
    return reshaped.view(torch.int32)


def native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits_u(x):
    x = x.view(torch.uint8).view(x.size(0), -1)
    unpacked = torch.stack([x & 15, x >> 4], dim=-1)
    reshaped = unpacked.view(x.size(0), -1)
    return reshaped.view(torch.int32)


disable_all_optimizations = False

if not hasattr(torch, 'uint16'):
    disable_all_optimizations = True

if disable_all_optimizations:
    print('You are using PyTorch below version 2.3. Some optimizations will be disabled.')

if not disable_all_optimizations:
    native_4bits_lookup_table = native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits(torch.arange(start=0, end=256*256, dtype=torch.long).to(torch.uint16))[:, 0]
    native_4bits_lookup_table_u = native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits_u(torch.arange(start=0, end=256*256, dtype=torch.long).to(torch.uint16))[:, 0]


def quick_unpack_4bits(x):
    if disable_all_optimizations:
        return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1).view(torch.int8) - 8

    global native_4bits_lookup_table

    s0 = x.size(0)
    x = x.view(torch.uint16)

    if native_4bits_lookup_table.device != x.device:
        native_4bits_lookup_table = native_4bits_lookup_table.to(device=x.device)

    y = torch.index_select(input=native_4bits_lookup_table, dim=0, index=x.to(dtype=torch.int32).flatten())
    y = y.view(torch.int8)
    y = y.view(s0, -1)

    return y


def quick_unpack_4bits_u(x):
    if disable_all_optimizations:
        return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1)

    global native_4bits_lookup_table_u

    s0 = x.size(0)
    x = x.view(torch.uint16)

    if native_4bits_lookup_table_u.device != x.device:
        native_4bits_lookup_table_u = native_4bits_lookup_table_u.to(device=x.device)

    y = torch.index_select(input=native_4bits_lookup_table_u, dim=0, index=x.to(dtype=torch.int32).flatten())
    y = y.view(torch.uint8)
    y = y.view(s0, -1)

    return y


def change_4bits_order(x):
    y = torch.stack([x & 15, x >> 4], dim=-2).view(x.size(0), -1)
    z = y[:, ::2] | (y[:, 1::2] << 4)
    return z
