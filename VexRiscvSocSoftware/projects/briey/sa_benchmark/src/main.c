// src/main.c
#include <stdint.h>
#include <stddef.h>    // <--- added: defines size_t
#include <briey.h>
#include <stdio.h>
#include <stdlib.h>

// keep your existing defines
#define DIM 768
#define MEM_SIZE 18

// ==========================================
// Cycle Counter
// ==========================================
static inline uint32_t rdcycle(void) {
    uint32_t cycles;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// ==========================================
// UART Helper
// ==========================================
void print(const char *str) {
    while(*str) {
        uart_write(UART, *(str++));
    }
}

void print_hex(uint32_t val) {
    const char hex[] = "0123456789ABCDEF";
    print("0x");
    for(int i = 28; i >= 0; i -= 4) {
        uart_write(UART, hex[(val >> i) & 0xF]);
    }
}

// ==========================================
// Software Math Functions
// ==========================================
float my_expf(float x) {
    if (x > 10.0f) return 88.0f;
    if (x < -10.0f) return 0.0f;

    float result = 1.0f;
    float term = 1.0f;
    for(int i = 1; i < 10; i++) {
        term *= x / (float)i;
        result += term;
    }
    return result;
}

float my_fmaxf(float a, float b) {
    return (a > b) ? a : b;
}

// ==========================================
// Big Arrays (go to SDRAM via .bss)
// ==========================================
static float x[DIM];
static float state[5 * MEM_SIZE * DIM];
static float time_mix_k[DIM];
static float time_mix_v[DIM];
static float time_mix_r[DIM];
static float time_first[DIM];
static float time_decay[DIM];
static float kw[DIM * DIM];
static float vw[DIM * DIM];
static float rw[DIM * DIM];
static float ow[DIM * DIM];

// Temporary arrays
static float xk[DIM], xv[DIM], xr[DIM];
static float r_lin[DIM], r[DIM], k[DIM], v[DIM];
static float kk[DIM], vv[DIM], aa[DIM], bb[DIM], pp[DIM];
static float ww[DIM], p[DIM], e1[DIM], e2[DIM];
static float a[DIM], b[DIM], wkv[DIM], r_wkv[DIM], out[DIM];

// ==========================================
// Helper Functions
// ==========================================
void sigmoid_vec(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float ex = my_expf(-input[i]);
        output[i] = 1.0f / (1.0f + ex);
    }
}

void matvec_mul(const float* mat, const float* vec, float* out_vec, int dim) {
    for (int i = 0; i < dim; i++) {
        float acc = 0.0f;
        const float* row = mat + ((size_t)i * dim);
        for (int j = 0; j < dim; j++) {
            acc += row[j] * vec[j];
        }
        out_vec[i] = acc;
    }
}

// ==========================================
// SA Function (Self-Attention)
// ==========================================
void SA(int i) {
    float* state_row_x  = &state[(5*i + 1) * DIM];
    float* state_row_aa = &state[(5*i + 2) * DIM];
    float* state_row_bb = &state[(5*i + 3) * DIM];
    float* state_row_pp = &state[(5*i + 4) * DIM];

    // 1) Time-mix
    for (int idx = 0; idx < DIM; idx++) {
        float s_val = state_row_x[idx];
        xk[idx] = x[idx] * time_mix_k[idx] + s_val * (1.0f - time_mix_k[idx]);
        xv[idx] = x[idx] * time_mix_v[idx] + s_val * (1.0f - time_mix_v[idx]);
        xr[idx] = x[idx] * time_mix_r[idx] + s_val * (1.0f - time_mix_r[idx]);
        state_row_x[idx] = x[idx];
    }

    // 2) Linear projections
    matvec_mul(rw, xr, r_lin, DIM);
    sigmoid_vec(r_lin, r, DIM);
    matvec_mul(kw, xk, k, DIM);
    matvec_mul(vw, xv, v, DIM);

    // 3) Copy k,v
    for (int idx = 0; idx < DIM; idx++) {
        kk[idx] = k[idx];
        vv[idx] = v[idx];
    }

    // 4) Load state
    for (int idx = 0; idx < DIM; idx++) {
        aa[idx] = state_row_aa[idx];
        bb[idx] = state_row_bb[idx];
        pp[idx] = state_row_pp[idx];
    }

    // 5) Compute a,b
    for (int idx = 0; idx < DIM; idx++) {
        ww[idx] = time_first[idx] + kk[idx];
        p[idx]  = my_fmaxf(pp[idx], ww[idx]);
        e1[idx] = my_expf(pp[idx] - p[idx]);
        e2[idx] = my_expf(ww[idx] - p[idx]);
        a[idx]  = e1[idx] * aa[idx] + e2[idx] * vv[idx];
        b[idx]  = e1[idx] * bb[idx] + e2[idx];
    }

    // 6) Update state
    for (int idx = 0; idx < DIM; idx++) {
        ww[idx] = pp[idx] + time_decay[idx];
        p[idx]  = my_fmaxf(ww[idx], kk[idx]);
        e1[idx] = my_expf(ww[idx] - p[idx]);
        e2[idx] = my_expf(kk[idx] - p[idx]);
        state_row_aa[idx] = e1[idx] * aa[idx] + e2[idx] * vv[idx];
        state_row_bb[idx] = e1[idx] * bb[idx] + e2[idx];
        state_row_pp[idx] = p[idx];
    }

    // 7) wkv = a / b
    for (int idx = 0; idx < DIM; idx++) {
        float denom = b[idx];
        if (denom == 0.0f) denom = 1e-9f;
        wkv[idx] = a[idx] / denom;
    }

    // 8) r * wkv
    for (int idx = 0; idx < DIM; idx++) {
        r_wkv[idx] = r[idx] * wkv[idx];
    }

    // 9) out = ow @ (r * wkv)
    matvec_mul(ow, r_wkv, out, DIM);
}

// ==========================================
// Initialize Arrays
// ==========================================
void init_arrays(void) {
    print("Initializing x, time_mix...\n");
    for(int i = 0; i < DIM; i++) {
        x[i] = 0.1f * (i % 100);
        time_mix_k[i] = 0.5f;
        time_mix_v[i] = 0.5f;
        time_mix_r[i] = 0.5f;
        time_first[i] = 0.1f;
        time_decay[i] = 0.01f;
    }

    print("Initializing weight matrices...\n");
    for(int i = 0; i < DIM * DIM; i++) {
        kw[i] = 0.01f;
        vw[i] = 0.01f;
        rw[i] = 0.01f;
        ow[i] = 0.01f;
    }

    print("Initializing state...\n");
    for(int i = 0; i < 5 * MEM_SIZE * DIM; i++) {
        state[i] = 0.0f;
    }

    print("Initialization complete!\n");
}

// ==========================================
// Main
// ==========================================
int main(void) {
    uint32_t start, end, cycles;

    // Configure UART
    Uart_Config uartConfig;
    uartConfig.dataLength = 8;
    uartConfig.parity = NONE;
    uartConfig.stop = ONE;
    uartConfig.clockDivider = 50000000/8/115200-1;
    uart_applyConfig(UART, &uartConfig);

    print("\n\n");
    print("========================================\n");
    print("  SA Benchmark - DIM=768, MEM_SIZE=18\n");
    print("========================================\n");
    print("Memory Layout:\n");
    print("  Code:   SDRAM (0x40000000)\n");
    print("  Arrays: SDRAM (0x40000000 + offset)\n");
    print("  Stack:  On-chip RAM (0x80000000)\n");
    print("========================================\n\n");

    // Initialize data
    print("Phase 1: Initializing arrays...\n");
    init_arrays();
    print("Done!\n\n");

    // Run SA
    print("Phase 2: Running SA (layer 0)...\n");

    start = rdcycle();
    SA(0);
    end = rdcycle();

    cycles = end - start;

    print("Done!\n\n");

    // Print results
    print("========================================\n");
    print("Results:\n");
    print("  Output[0]:   ");
    print_hex(*(uint32_t*)&out[0]);
    print("\n  Output[767]: ");
    print_hex(*(uint32_t*)&out[767]);
    print("\n  Cycles:      ");
    print_hex(cycles);
    print("\n========================================\n\n");

    // Hang
    while(1);
}

void irqCallback(void) {
    // Not used
}

