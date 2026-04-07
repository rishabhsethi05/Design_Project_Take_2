#include <stdio.h>

unsigned int compute_crc(unsigned char *data, int length) {
    unsigned int crc = 0xFFFFFFFF;

    for (int i = 0; i < length; i++) {
        crc ^= data[i];

        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320;
            else
                crc = crc >> 1;
        }
    }

    return crc;
}