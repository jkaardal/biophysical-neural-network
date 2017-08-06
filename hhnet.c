#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#define p3(x) x*x*x
#define p4(x) x*x*x*x
#define eps 1.0E-10
#define PROFILE 0

typedef struct {
    // general constants
    double Cm;  // membrane capacitance (uF/cm^2)
    double VNa; // Na reversal potential (mV)
    double VK;  // K reversal potential (mV)
    double VL;  // "leakage" potential (mV)
    double gNa; // Na channel conductance (m/Ohm*cm^2)
    double gK;  // K channel conductance (m/Ohm*cm^2)
    double gL;  // "leakage" conductance (m/Ohm*cm^2)
    // n activation constants: activation(V) = (zn[0] + zn[1]*V)/(zn[2] + zn[3]*exp((zn[4]+V)/zn[5]))
    double an[6];
    double bn[6];
    // m activation
    double am[6];
    double bm[6];
    // h activation
    double ah[6];
    double bh[6];

    // state variables
    double *V;    // membrane potential difference (mV); if time series V[0] is t=0.0, V[1] is t=-dt, V[2] is t=-2*dt, etc.
    double *n;    // ionic activation 1; same as above for time series
    double *m;    // ionic activation 2; same as above for time series
    double *h;    // ionic activation 3; same as above for time series
    double Iext; // external current (uA/cm^2)
    int NV;
    int Nn;
    int Nm;
    int Nh;

    // position coordinates (if applicable)
    double *r;
} neuron;

double activ(double *z, double v) {
    // rate "constants" of an ion channel
    return (z[0] + z[1]*v)/(z[2] + z[3]*exp((z[4]+v)/z[5]));
}

double igrad(double x, double a, double b) {
    // ion channel activation/inactivation rates
    return a*(1.0-x)-b*x;
}

double Vgrad(neuron *x) {
    // change in membrane voltage per unit time (r.h.s. of differential equation)
    return -(x->gK*p4(x->n[0])*(x->V[1] - x->VK) + x->gNa*p3(x->m[0])*x->h[0]*(x->V[1] - x->VNa) + x->gL*(x->V[1] - x->VL) + x->Iext)/x->Cm;
}

void init_activ_state(neuron *x) {
    // initialize ion channel activations
    x->n[0] = activ(x->an, x->V[0])/(activ(x->an, x->V[0]) + activ(x->bn, x->V[0]));
    x->m[0] = activ(x->am, x->V[0])/(activ(x->am, x->V[0]) + activ(x->bm, x->V[0]));
    x->h[0] = activ(x->ah, x->V[0])/(activ(x->ah, x->V[0]) + activ(x->bh, x->V[0]));
}

void init_lit(neuron *x) {
    // initialize physical quantities to literature values
    x->Cm = 1.0;
    x->VNa = -115.0;
    x->VK = 12.0;
    x->VL = 10.613;
    x->gNa = 120.0;
    x->gK = 36.0;
    x->gL = 0.3;

    x->an[0] = 0.1; x->an[1] = 0.01; x->an[2] = -1.0; x->an[3] = 1.0; x->an[4] = 10.0; x->an[5] = 10.0;
    x->bn[0] = 0.125; x->bn[1] = 0.0; x->bn[2] = 0.0; x->bn[3] = 1.0; x->bn[4] = 0.0; x->bn[5] = -80.0;

    x->am[0] = 2.5; x->am[1] = 0.1; x->am[2] = -1.0; x->am[3] = 1.0; x->am[4] = 25.0; x->am[5] = 10.0;
    x->bm[0] = 4.0; x->bm[1] = 0.0; x->bm[2] = 0.0; x->bm[3] = 1.0; x->bm[4] = 0.0; x->bm[5] = -18.0;

    x->ah[0] = 0.07; x->ah[1] = 0.0; x->ah[2] = 0.0; x->ah[3] = 1.0; x->ah[4] = 0.0; x->ah[5] = -20.0;
    x->bh[0] = 1.0; x->bh[1] = 0.0; x->bh[2] = 1.0; x->bh[3] = 1.0; x->bh[4] = 30.0; x->bh[5] = 10.0;
}

void zero_currents(neuron *x, int N) {
    // set external currents to zero
    int i;

    for(i=0; i<N; i++)
        x[i].Iext = 0.0;
}

void leapfrog(neuron *x, int N, double *t, double dt, double T, void (**I)(neuron*, int, double, double*, int*, char*), int M, double *args, int *iargs, char *cargs, int *offsets) {
    int i;
    // leapfrog integration of the coupled Hodgkin-Huxley model differential equations of a neural network

    // calculate external currents here for step t
    if(PROFILE)
        printf("Calculating stimulus\n");
    for(i=0; i<M; i++)
        (*I[i])(x, N, *t, args+offsets[i*3], iargs+offsets[i*3+1], cargs+offsets[i*3+2]);
    if(PROFILE)
        printf("Integrating\n");
    for(i=0; i<N; i++) {
        // half-step n, m, h
        x[i].n[0] = x[i].n[1] + 0.5*dt*igrad(x[i].n[1], activ(x[i].an, x[i].V[1]), activ(x[i].bn, x[i].V[1]));
        x[i].m[0] = x[i].m[1] + 0.5*dt*igrad(x[i].m[1], activ(x[i].am, x[i].V[1]), activ(x[i].bm, x[i].V[1]));
        x[i].h[0] = x[i].h[1] + 0.5*dt*igrad(x[i].h[1], activ(x[i].ah, x[i].V[1]), activ(x[i].bh, x[i].V[1]));

        // full-step V
        x[i].V[0] = x[i].V[1] + dt*Vgrad(x+i);

        // half-step n, m, h
        x[i].n[0] += 0.5*dt*igrad(x[i].n[0], activ(x[i].an, x[i].V[0]), activ(x[i].bn, x[i].V[0]));
        x[i].m[0] += 0.5*dt*igrad(x[i].m[0], activ(x[i].am, x[i].V[0]), activ(x[i].bm, x[i].V[0]));
        x[i].h[0] += 0.5*dt*igrad(x[i].h[0], activ(x[i].ah, x[i].V[0]), activ(x[i].bh, x[i].V[0]));
    }

    (*t) += dt;
}

int maxint(int *x, int N) {
    // find the maximum integer in array x of length N
    int i, max;

    max = x[0];
    for(i=1; i<N; i++) {
        if(x[i] > max)
            max = x[i];
    }

    return max;
}

// external stimuli
void hit_stim(neuron *x, int N, double t, double *args, int *iargs, char *cargs) {
    // sudden impulse of external current
    int i;

    if(t == 0.0) {
        for(i=0; i<N; i++) {
            x[i].Iext += 2000.0;
        }
    }
}

void osc_stim(neuron *x, int N, double t, double *args, int *iargs, char *cargs) {
    // oscillatory external current
    int i;

    for(i=0; i<N; i++) {
        if(cargs[i] == 1) {
            x[i].Iext += args[i*3]*sin(2.0*M_PI*t/args[i*3+1] + args[i*3+2])*sin(2.0*M_PI*t/args[i*3+1] + args[i*3+2]);
        }
    }
}

// neural network internal stimulation
void trans_net(neuron *x, int N, double t, double *args, int *iargs, char *cargs) {
    // input the neurons in the network from connections to other neurons
    int i, j;
    double aIN, aOUT, bIN, bOUT, g;
    double Vi, Vj;
    int *nhist = iargs;
    double *C = args;

    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            aIN = C[5*(i*N+j)];
            aOUT = C[5*(i*N+j)+1];
            bIN = C[5*(i*N+j)+2];
            bOUT = C[5*(i*N+j)+3];
            g = C[5*(i*N+j)+4];
            Vi = x[i].V[1];
            Vj = x[j].V[nhist[i*N+j]+1];
            //printf("%f, %f, %f, %f, %f, %f, %f\n", aIN, aOUT, bIN, bOUT, g, Vi, Vj);
            x[i].Iext += g/(1.0 + exp(aIN+bIN*Vj))/(1.0 + exp(-aOUT-bOUT*Vi))/(1.0 + exp(-aOUT+bOUT*Vi));
        }
    }
}

void circshift(double *x, int N, int d) {
    // circularly shift x of length N forward by d elements
    if(d == 0) {
        return;
    }

    double *y = malloc(N * sizeof *y);
    if(d > 0) {
        memcpy(y+d, x, (N-d) * sizeof *x);
        memcpy(y, x+N-d, d * sizeof *x);
    }
    else {
        memcpy(y+N-d, x, d * sizeof *x);
        memcpy(y, x+d, (N-d) * sizeof *x);
    }
    memcpy(x, y, N * sizeof *y);

    free(y);
}

void calloc_neuron(neuron *x, int NV, int Nn, int Nm, int Nh) {
    // allocate a neuron's voltage and activation arrays and initialize to zero
    x->V = calloc(NV, sizeof *(x->V));
    x->n = calloc(Nn, sizeof *(x->n));
    x->m = calloc(Nm, sizeof *(x->m));
    x->h = calloc(Nh, sizeof *(x->h));
    x->NV = NV;
    x->Nn = Nn;
    x->Nm = Nm;
    x->Nh = Nh;
}

void malloc_neuron(neuron *x, int NV, int Nn, int Nm, int Nh) {
    // allocate a neuron's voltage and activation arrays 
    x->V = malloc(NV * sizeof *(x->V));
    x->n = malloc(Nn * sizeof *(x->n));
    x->m = malloc(Nm * sizeof *(x->m));
    x->h = malloc(Nh * sizeof *(x->h));
    x->NV = NV;
    x->Nn = Nn;
    x->Nm = Nm;
    x->Nh = Nh;
}

void free_neurons(neuron *x, int N) {
    // free the memory of a neuron's voltage and activation arrays
    int i;

    for(i=0; i<N; i++) {
        free(x[i].V);
        free(x[i].n);
        free(x[i].m);
        free(x[i].h);
    }
    free(x);
}

double randn() {
    // generate a normally distributed random number
    double X1; //, X2;
    double Y1, Y2;

    do {
        Y1 = 1.0*rand()/RAND_MAX;
        Y2 = 1.0*rand()/RAND_MAX;
    } while(Y1 <= eps);

    X1 = sqrt(-2.0*log(Y1))*cos(2.0*M_PI*Y2);
    //X2 = sqrt(-2.0*log(Y1))*sin(2.0*M_PI*Y2);

    return X1;
}

double round(double x) {
    // round a number to the nearest integer
    return ( (x-floor(x)) >= 0.5 ? ceil(x) : floor(x) ); 
}

// single neuron dynamics
/*int main(int argc, char *argv[]) {
    neuron *x = malloc(1 * sizeof *x);
    double t = 0.0;   // starting time (ms)
    double dt = 0.01; // time step (ms)
    double T = 20.0;  // termination time (ms)
    void (*I)(neuron*, int, double, double*, int*, char*) = &osc_stim; // external current function
    double *args = malloc(3 * sizeof *args);    // double precision arguments for I
    int *iargs = NULL;                          // integer precision arguments for I
    char *cargs = malloc(1 * sizeof *cargs);    // character precision arguments for I
    int *offsets = calloc(0, 3 * sizeof *offsets); // offsets of argument arrays for each I function

    calloc_neuron(x, 2, 2, 2, 2);
    init_lit(x);
    init_activ_state(x);

    args[0] = 100.0; args[1] = 16.0; args[2] = 0.0;
    cargs[0] = 1;

    printf("%f, %f\n", t, -x->V[0]);
    circshift(x->V, x->NV, 1);
    circshift(x->n, x->Nn, 1);
    circshift(x->m, x->Nm, 1);
    circshift(x->h, x->Nh, 1);
    while(t <= T) {
        zero_currents(x, 1);
        leapfrog(x, 1, &t, dt, T, &I, 1, args, iargs, cargs, offsets);
        printf("%f, %f\n", t, -x->V[0]);
        circshift(x->V, x->NV, 1);
        circshift(x->n, x->Nn, 1);
        circshift(x->m, x->Nm, 1);
        circshift(x->h, x->Nh, 1);
    }

    free(args);
    free(iargs);
    free(cargs);
    free(offsets);
    free_neurons(x, 1);

    return 0;
}*/

// two neurons, stable dynamics
/*int main(int argc, char *argv[]) {
    int N = 2;
    double t = 0.0;
    double dt = 0.01;
    double T = 1000.0;
    void (**I)(neuron*, int, double, double*, int*, char*);
    double *args;
    int *iargs;
    char *cargs;
    int *offsets;
    int max_t;

    int i, j;

    neuron *x = malloc(N * sizeof *x);

    // transmission delays
    int *nhist = calloc(N*N, sizeof *nhist);
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            if(i == j)
                continue;
            nhist[i*N+j] = 500;
        }
    }
    max_t = maxint(nhist, N*N);
    // initialize neurons (with membrane potential history)
    for(i=0; i<N; i++) {
        malloc_neuron(x+i, max_t+2, 2, 2, 2);
        x[i].V[0] = 0.0;
        init_lit(x+i);
        init_activ_state(x+i);
        for(j=1; j<(max_t+2); j++) {
            x[i].V[j] = x[i].V[0];
        }
        x[i].n[1] = x[i].n[0];
        x[i].m[1] = x[i].m[0];
        x[i].h[1] = x[i].h[0];
    }
    
    I = malloc(2 * sizeof (*I));
    I[0] = osc_stim;  // stimulation function
    I[1] = trans_net; // network transmission function

    // allocate memory for optional arguments
    args = malloc((3+(5*N)*(5*N)) * sizeof *args);
    iargs = malloc(N*N * sizeof *args);
    cargs = malloc(N * sizeof *cargs);
    offsets = malloc(3*N * sizeof *offsets);

    // stimulation arguments (only first neuron stimulated)
    offsets[0] = 0; offsets[1] = 0; offsets[2] = 0;
    args[0] = 100.0; // amplitude
    args[1] = 16.0;  // wavelength denom.
    args[2] = 0.0;   // phase
    cargs[0] = 1;    // stimulation input 1 (true) or 0 (false)
    memset(cargs+1, 0, (N-1) * sizeof *cargs);

    // transmission arguments
    offsets[3] = 3; offsets[4] = 0; offsets[5] = N;
    memcpy(iargs, nhist, N*N * sizeof *nhist);
    free(nhist);
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            args[offsets[3] + 5*(i*N + j)] = 50.0;
            args[offsets[3] + 5*(i*N + j) + 1] = 75.0;
            args[offsets[3] + 5*(i*N + j) + 2] = 1.0;
            args[offsets[3] + 5*(i*N + j) + 3] = 1.0;
            if(i == j)
                args[offsets[3] + 5*(i*N + j) + 4] = 0.0;
            else
                args[offsets[3] + 5*(i*N + j) + 4] = 20.0;
        }
    }

    printf("%f, ", t);
    for(i=0; i<(N-1); i++) {
        printf("%f, ", -x[i].V[0]);
        circshift(x[i].V, x[i].NV, 1);
        circshift(x[i].n, x[i].Nn, 1);
        circshift(x[i].m, x[i].Nm, 1);
        circshift(x[i].h, x[i].Nh, 1);
    }
    printf("%f\n", -x[N-1].V[0]);
    circshift(x[N-1].V, x[N-1].NV, 1);
    circshift(x[N-1].n, x[N-1].Nn, 1);
    circshift(x[N-1].m, x[N-1].Nm, 1);
    circshift(x[N-1].h, x[N-1].Nh, 1);
    while(t <= T) {
        zero_currents(x, N);
        leapfrog(x, N, &t, dt, T, I, 2, args, iargs, cargs, offsets);
        printf("%f, ", t);
        for(i=0; i<(N-1); i++) {
            printf("%f, ", -x[i].V[0]);
            circshift(x[i].V, x[i].NV, 1);
            circshift(x[i].n, x[i].Nn, 1);
            circshift(x[i].m, x[i].Nm, 1);
            circshift(x[i].h, x[i].Nh, 1);
        }
        printf("%f\n", -x[N-1].V[0]);
        circshift(x[N-1].V, x[N-1].NV, 1);
        circshift(x[N-1].n, x[N-1].Nn, 1);
        circshift(x[N-1].m, x[N-1].Nm, 1);
        circshift(x[N-1].h, x[N-1].Nh, 1);
    }

    free(args);
    free(iargs);
    free(cargs);
    free(offsets);
    free(I);
    free_neurons(x, N);

    return 0;
}*/

// multiple neurons in a network
int main(int argc, char *argv[]) {
    int N = 1; //100;
    double t = 0.0;
    double dt = 0.01;
    double T = 20.0; // (ms)
    int skip = 0;
    void (**I)(neuron*, int, double, double*, int*, char*);
    double *args;
    int *iargs;
    char *cargs;
    int *offsets;
    int max_t;
    double gain = 550.0;

    const char filename[] = "./data.txt";
    FILE *fp;
    fp = fopen(filename, "w");
    if(fp == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }
    fclose(fp);

    int i, j, iter=0;

    srand(time(NULL));

    neuron *x = malloc(N * sizeof *x);

    // transmission delays
    int *nhist = calloc(N*N, sizeof *nhist);
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            if(i == j)
                continue;
            nhist[i*N+j] = (int)round((1.0*rand()/RAND_MAX)*500);
        }
    }
    max_t = maxint(nhist, N*N);
    // initialize neurons (with membrane potential history)
    for(i=0; i<N; i++) {
        malloc_neuron(x+i, max_t+2, 2, 2, 2);
        x[i].V[0] = 0.0;
        init_lit(x+i);
        init_activ_state(x+i);
        for(j=1; j<(max_t+2); j++) {
            x[i].V[j] = x[i].V[0];
        }
        x[i].n[1] = x[i].n[0];
        x[i].m[1] = x[i].m[0];
        x[i].h[1] = x[i].h[0];
    }
    
    I = malloc(2 * sizeof (*I));
    I[0] = osc_stim;  // stimulation function
    I[1] = trans_net; // network transmission function

    // allocate memory for optional arguments
    args = malloc((3+(5*N)*(5*N)) * sizeof *args);
    iargs = malloc(N*N * sizeof *args);
    cargs = malloc(N * sizeof *cargs);
    offsets = malloc(3*N * sizeof *offsets);

    // stimulation arguments (only first neuron stimulated)
    offsets[0] = 0; offsets[1] = 0; offsets[2] = 0;
    args[0] = 100.0; // amplitude
    args[1] = 16.0;  // wavelength denom.
    args[2] = 0.0;   // phase
    cargs[0] = 1;    // stimulation input 1 (true) or 0 (false)
    memset(cargs+1, 0, (N-1) * sizeof *cargs);

    // transmission arguments
    offsets[3] = 3; offsets[4] = 0; offsets[5] = N;
    memcpy(iargs, nhist, N*N * sizeof *nhist);
    free(nhist);
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            args[offsets[3] + 5*(i*N + j)] = 50.0;
            args[offsets[3] + 5*(i*N + j) + 1] = 75.0;
            args[offsets[3] + 5*(i*N + j) + 2] = 1.0;
            args[offsets[3] + 5*(i*N + j) + 3] = 1.0;
            if(i == j)
                args[offsets[3] + 5*(i*N + j) + 4] = 0.0;
            else {
                args[offsets[3] + 5*(i*N + j) + 4] = gain*randn()/N;
            }
        }
    }

    fp = fopen(filename, "a");
    if(fp == NULL) {
        printf("Error opening file.\n");
        goto FREEALL;
    }
    fprintf(fp, "%f, ", t);
    //printf("%f, ", t);
    for(i=0; i<(N-1); i++) {
        fprintf(fp, "%f, ", -x[i].V[0]);
        //printf("%f, ", -x[i].V[0]);
        circshift(x[i].V, x[i].NV, 1);
        circshift(x[i].n, x[i].Nn, 1);
        circshift(x[i].m, x[i].Nm, 1);
        circshift(x[i].h, x[i].Nh, 1);
    }
    fprintf(fp, "%f\n", -x[i].V[0]);
    //printf("%f\n", -x[N-1].V[0]);
    circshift(x[N-1].V, x[N-1].NV, 1);
    circshift(x[N-1].n, x[N-1].Nn, 1);
    circshift(x[N-1].m, x[N-1].Nm, 1);
    circshift(x[N-1].h, x[N-1].Nh, 1);
    while(t <= T) {
        if(PROFILE)
            printf("Starting step at t=%f\n", t);
        zero_currents(x, N);
        //if(t > 10.0)
        //    cargs[0] = 0;
        leapfrog(x, N, &t, dt, T, I, 2, args, iargs, cargs, offsets);
        if(iter >= skip)
            fprintf(fp, "%f, ", t);
        //printf("%f, ", t);
        for(i=0; i<(N-1); i++) {
            if(iter >= skip)
                fprintf(fp, "%f, ", -x[i].V[0]);
            //printf("%f, ", -x[i].V[0]);
            circshift(x[i].V, x[i].NV, 1);
            circshift(x[i].n, x[i].Nn, 1);
            circshift(x[i].m, x[i].Nm, 1);
            circshift(x[i].h, x[i].Nh, 1);
        }
        if(iter >= skip) {
            fprintf(fp, "%f\n", -x[N-1].V[0]);
            iter = 0;
        }
        else
            iter += 1;
        //printf("%f\n", -x[N-1].V[0]);
        circshift(x[N-1].V, x[N-1].NV, 1);
        circshift(x[N-1].n, x[N-1].Nn, 1);
        circshift(x[N-1].m, x[N-1].Nm, 1);
        circshift(x[N-1].h, x[N-1].Nh, 1);
    }
    fclose(fp);

    FREEALL:
    free(args);
    free(iargs);
    free(cargs);
    free(offsets);
    free(I);
    free_neurons(x, N);

    return 0;
}




