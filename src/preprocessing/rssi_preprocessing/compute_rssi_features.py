"""
Script provided by original hosts of the the competition. Can be found at:
https://github.com/location-competition/indoor-location-competition-20/blob/master/main.py
"""

import numpy as np

from src.preprocessing.imu_preprocessing.step_relative_position_extractor import split_ts_seq


def calibrate_magnetic_wifi_ibeacon_to_position(
    magn_datas, wifi_datas, ibeacon_datas, step_positions
):
    mwi_datas = {}

    if wifi_datas.size != 0:
        sep_tss = np.unique(wifi_datas[:, 0].astype(float))
        wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
        for wifi_ds in wifi_datas_list:
            diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]["wifi"] = np.append(
                    mwi_datas[target_xy_key]["wifi"], wifi_ds, axis=0
                )
            else:
                mwi_datas[target_xy_key] = {
                    "magnetic": np.zeros((0, 4)),
                    "wifi": wifi_ds,
                    "ibeacon": np.zeros((0, 3)),
                }

    if ibeacon_datas.size != 0:
        sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
        ibeacon_datas_list = split_ts_seq(ibeacon_datas, sep_tss)
        for ibeacon_ds in ibeacon_datas_list:
            diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]["ibeacon"] = np.append(
                    mwi_datas[target_xy_key]["ibeacon"], ibeacon_ds, axis=0
                )
            else:
                mwi_datas[target_xy_key] = {
                    "magnetic": np.zeros((0, 4)),
                    "wifi": np.zeros((0, 5)),
                    "ibeacon": ibeacon_ds,
                }

    sep_tss = np.unique(magn_datas[:, 0].astype(float))
    magn_datas_list = split_ts_seq(magn_datas, sep_tss)
    for magn_ds in magn_datas_list:
        diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
        index = np.argmin(diff)
        target_xy_key = tuple(step_positions[index, 1:3])
        if target_xy_key in mwi_datas:
            mwi_datas[target_xy_key]["magnetic"] = np.append(
                mwi_datas[target_xy_key]["magnetic"], magn_ds, axis=0
            )
        else:
            mwi_datas[target_xy_key] = {
                "magnetic": magn_ds,
                "wifi": np.zeros((0, 5)),
                "ibeacon": np.zeros((0, 3)),
            }

    return mwi_datas


def extract_magnetic_strength(mwi_datas):
    magnetic_strength = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        magnetic_data = mwi_datas[position_key]["magnetic"]
        magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_strength[position_key] = magnetic_s

    return magnetic_strength


def extract_wifi_rssi(mwi_datas):
    wifi_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]["wifi"]
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])

            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (
                        old_count + 1
                    )
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            wifi_rssi[bssid] = position_rssi

    return wifi_rssi


def extract_ibeacon_rssi(mwi_datas):
    ibeacon_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        ibeacon_data = mwi_datas[position_key]["ibeacon"]
        for ibeacon_d in ibeacon_data:
            ummid = ibeacon_d[1]
            rssi = int(ibeacon_d[2])

            if ummid in ibeacon_rssi:
                position_rssi = ibeacon_rssi[ummid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (
                        old_count + 1
                    )
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            ibeacon_rssi[ummid] = position_rssi

    return ibeacon_rssi


def extract_wifi_count(mwi_datas):
    wifi_counts = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]["wifi"]
        count = np.unique(wifi_data[:, 2]).shape[0]
        wifi_counts[position_key] = count

    return wifi_counts
