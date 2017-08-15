#include "attitude_estimator.h"

using namespace stateestimation;

extern "C" {

    AttitudeEstimator est;
    double q[4];

    void init_est() {
        est.setGyroBias(0.0, 0.0, 0.0);
        est.reset(true, false);
    }

    void update_est(double dt, double g0, double g1, double g2, double a0, double a1, double a2) {
        est.update(dt, g0, g1, g2, a0, a1, a2, 0, 0, 0);
        est.getAttitude(q);
    }

    double q0() { return q[0]; }
    double q1() { return q[1]; }
    double q2() { return q[2]; }
    double q3() { return q[3]; }

}