Skip to Main Content
IEEE.org
IEEE Xplore
IEEE SA
IEEE Spectrum
More Sites
Donate
Cart
Create Account
Personal Sign In
IEEE Xplore logo - Link to home
Browse 
My Settings 
Help 
Access provided by:
NANJING UNIVERSITY OF AERONAUTICS AND ASTRONAUTICS
Sign Out
IEEE logo - Link to IEEE main site homepage

All

ADVANCED SEARCH
Conferences >2014 UKACC International Conf...
Real-time obstacle collision avoidance for fixed wing aircraft using B-splines
Publisher: IEEE
Cite This
PDF
Hamid Alturbeh; James F. Whidborne
All Authors

11
Cites in
Papers

661
Full
Text Views

Abstract
Document Sections
I.
Introduction
II.
Fixed Wing Aircraft Model
III.
Trajectory Description
IV.
Local Trajectory Optimization
V.
Simulation Results
Show Full Outline
Authors
Figures
References
Citations
Keywords
Metrics
Abstract:
A real-time collision avoidance algorithm is developed based on parameterizing an optimal control problem with B-spline curves. The optimal control problem is formulated in output space rather than control or input space, this is feasible because ofthe differential flatness of the system for a fixed wing aircraft. The flat output trajectory is parameterized using a B-spline curve representation. In order to reduce the computational time of the optimal problem, the aircraft and obstacle constraints are augmented in the cost function using a penalty function method. The developed algorithm has been simulated and tested in MATLAB/Simulink.
Published in: 2014 UKACC International Conference on Control (CONTROL)
Date of Conference: 09-11 July 2014
Date Added to IEEE Xplore: 02 October 2014
Electronic ISBN:978-1-4799-5011-9
DOI: 10.1109/CONTROL.2014.6915125
Publisher: IEEE
Conference Location: Loughborough, UK
SECTION I.Introduction
Unmanned Aircraft Systems (UAS) are of increasing importance in the aerospace industry for both civilian and military applications due to their ability to complete dull, dirty and dangerous missions [1]. However, operation of Unmanned Aerial Vehicles (UAV's) in civil/non-segregated airspace is restricted by the policies of aviation authorities which require full compliance with rules and obligation that apply for manned aircraft [2]. Trajectory tracking and collision avoidance are issues that a UAV must deal with in a way that gives the UAV the ability to avoid conflict situations. Thus, any UAV that will be operated in civil/non-segregated airspace must be equipped with a collision avoidance system that has the ability to avoid conflict scenarios in full compliance with airspace traffic rules. Much research is being undertaken to enable the routine use of UAV's in all classes of airspace without the need for restrictive or specialized conditions of operation. The ASTRAEA program [3] is one example.

Trajectory planners can be divided into two main categories [4]; global planners which require good knowledge about the environment that the aircraft is going to fly in, and local trajectory planners which are algorithms that run continuously in order to allow the aircraft to deal with events that may happen during the flight. Many methods for generating trajectories that guarantee collision avoidance have been proposed in the literature. For example: predefined, protocol based [5], E-field [6], geometric [7] and automotive [8].

This paper presents an approach for generating collision avoidance trajectories based on B-spline curves. Essentially, a finite-horizon optimal control problem is periodically solved in real-time hence updating the aircraft trajectory to avoid obstacles and drive the aircraft to its global path. The proposed approach can be summarized as follows:

Given a global trajectory that the aircraft is required to follow, solve the optimal control problem
minU(t)∈UJ(U(t))(1)
View SourceRight-click on figure for MathML and additional features.subject to the aircraft dynamics constraints pair, (X˙=f(X,U),Y=g(X)), state constraint, X(t)∈X, and aircraft trajectory obstacles constraint, Y(t)∈Y, where U∈U is the control and J is a cost measured over a finite time horizon, t∈[t0,tf], that drives the local trajectory to the global trajectory.

The problem is solved by a direct method by inverting the dynamics, so the optimization is performed in the output space Y(t)∈Y, and parameterizing the trajectory by a spline function. The cost is augmented to maintain the constraints.

The generated local trajectory allows the UAV to track the global trajectory while avoiding any intruder or conflict scenarios that may occur. The local trajectory optimization is periodically solved on-line in a receding horizon approach to account for system uncertainties and obstacle changes.

In Section II, the system model is described. Section III discusses B-spline basis functions and curves. Section IV shows the formulation of the optimal problem to find the optimal local trajectories. Section V presents the simulation results.

SECTION II.Fixed Wing Aircraft Model
A fixed wing aircraft dynamic can be expressed by a point-mass model [9]:
⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢⎢x˙y˙z˙γ˙χ˙V˙⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢⎢VcosγcosχVcosγsinχVsinγ(g/V)(ncosϕ−cosχ)(g/V)(nsinϕ/cosγ)(T−D)/m−gsinγ⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥⎥(2)
View SourceRight-click on figure for MathML and additional features.where x,y,z are the aircraft center of gravity coordinates in earth axis, γ is the flight-path angle, χ is the heading angle, V is the aircraft speed, g is the gravity acceleration, ϕ is the bank angle, T is the thrust, D is the drag, m is the total mass, n=L/(mg) is the load factor and L is the total aircraft lift. The input and output vectors are defined respectively as:
U=[ϕ  T  n]TY=[x  y  z]T(3)
View SourceRight-click on figure for MathML and additional features.

In order to determine an optimal control trajectory for aircraft using direct methods, the optimal control problem is formulated in output space rather than control or input space. However, the output design space technique is only available when the system is differentially flat [10]. A system is differentially flat if its states and inputs can be expressed as functions of the output vector and its derivatives [10], [11]. Fortunately most fixed-wing aircraft systems can be considered as differentially flat system, the following discussion shows that the fixed wing aircraft possesses the property of flatness. Modifying (2) obtains:
V=x˙2+y˙2+z˙2−−−−−−−−−−√γ=arcsin(z˙/V)χ=arcsin(y˙/(Vcosγ))ϕ=arctan((χ˙Vcosγ)/(gcosγ+Vγ¨))n=(gcosγ+Vγ˙)/(gcosϕ)T=D+mV˙+mgsinγ(4)(5)(6)(7)(8)(9)
View SourceRight-click on figure for MathML and additional features.

The aerodynamic drag is given by [12]:
D=12ρSCDV2(10)
View SourceRight-click on figure for MathML and additional features.where CD=CD0+kC2I is the drag coefficient, CL2nmg/(ρSV2) is the lift coefficient, ρ is the air density, CD0 is the minimum drag coefficient of the aircraft and S is the wing area. It can be noticed from (4)–​(10) that the inputs and the states of the system can be expressed as functions of the output vector and its derivatives, hence the system is differentially flat. So the optimal problem can be formulated in output space rather than control space. Thus, it is useful to find a sufficient description for the output space (trajectory profiles in our case) which makes the optimal problem more tractable.

SECTION III.Trajectory Description
NURBS curves are used to describe the trajectory profiles. A NURBS curve is a vector-valued piecewise rational polynomial function. The pth degree NURBS curve is given by:
Ri,p(τ)P(τ)=∑i=0nRi,p(τ)Ci=wiNi,p(τ)Ci∑ni=0wiNi,p(τ);  a≤τ≤b(11)(12)
View SourceRight-click on figure for MathML and additional features.where Ri,p(τ) are rational basis functions. The analytical properties of Ri,p(τ) determine the geometric behavior of curves [13], wi are the weights, Ci are the control points, and Ni,p(τ) are the pth degree B-spline basis functions. There are many ways to represent B-spline basis functions, for computer implementation the recursive representation of B-spline basis functions is the most useful form [13]. Let U=[u0,  u1,  …,  um−1,  um] be a nondecreasing sequence of real numbers i.e, ui≤ux+1;i=0,1,…,m−1,ui called knots or breakpoints, and U is the knot vector that contain m+1 knots. So the ith B-spline basis function of p-degree (order p+1), denoted by Np,i(τ) is defined as:
Ni,0(τ)=Ni,p(τ)={10if ui≤τ<ui+1 otherwise τ−uiui+p−uiNi,p−1(τ)+ui+p+1−τui+p+1−ui+1Ni+1,p−1(τ)(13)(14)
View SourceRight-click on figure for MathML and additional features.and Ni,p(τ)=0 if τ is outside [ui,ui+p+1[. The degree of the basis function p, number of control point (n+1), and number of the knots (m+1) are related by m=n+p+1.

The knot vector can be realized in different forms, but it must be a nondecreasing sequence of real numbers. There are two types of knot vector, periodic and open, in two flavours, uniform and nonuniform [14]. In a uniform knot vector, individual knot values are evenly spaced. In practice, uniform knot vectors generally begin at zero and are incremented by 1 to some maximum value, or it can be normalized in a range between 0 and 1. A periodic uniform knot vector will give periodic uniform basis functions for which:
Ni,p(τ)=Ni−1,p(τ−1)=Ni+1,p(τ+1)(15)
View SourceRight-click on figure for MathML and additional features.Thus, each basis function is a translation of the other.

In an open uniform knot vector, the end knot values have multiplicity equal to the order of the B-spline basis functions p+1. NURBS basis functions have many useful properties [14]. For example, they are nonnegative, satisfy the portion of unity property, have a local support, remain in the convex hull of the control points, and all their derivatives exist in the interior of the knot span [ui,ui+p+1[, where they are rational functions with nonzero denominators. The recursive calculation of the NURBS basis functions makes them easily, efficiently, and accurately processed in a computer. In particular:

the computation of point and derivatives on the curves is efficient;

they are numerical insensitive to floating point rounding error, and

they require little memory for storage requirements.

A. Derivatives of b-Spline Curves
The derivatives of B-spline curves can be calculated simply by computing the derivatives of their B-spline basis functions. The kth derivative of P(τ),P(k)(τ), is given by:
P(k)(τ)=∑i=0nN(k)i,p(τ)Ci(16)
View SourceRight-click on figure for MathML and additional features.where N(k)i,p(τ) is the kth derivative of B-spline basis functions which can be calculated recursively:
N(k)i,p(τ)=p⎛⎝N(k−1)i,p−1(τ)ui+p−ui−N(k−1)i+1,p−1(τ)ui+p+1−ui+1⎞⎠(17)
View SourceRight-click on figure for MathML and additional features.

SECTION IV.Local Trajectory Optimization
The optimal local trajectory profiles can be achieved by finding values of design variables that minimize a defined cost function and satisfy all constraints.

Fig. 1. - Bezier curve (top), and its basis functions (bottom)
Fig. 1.
Bezier curve (top), and its basis functions (bottom)

Show All

A. Bezier Curve
Bezier curves represent a special case of NURBS where all the weights are equal to unity, i.e. wi=1, and the knot vector is U=[0,0,0,0,0,0,0,1,1,1,1,1,1,1] (for p=6). In this case the basis functions are called Bernstein basis functions. A 6th order Bezier curve (p=6) has been used to represent the aircraft local trajectories. Using (12) and (14), the 6th order Bezier curve basis functions are
R0=(1−τ)6,R1=6τ(1−τ)5,R2ζ=15τ2(1−τ)4,R3=20τ3(1−τ)3,R4=15τ4(1−τ)2,R5=6τ5(1−τ),R6=τ6(18)
View SourceRight-click on figure for MathML and additional features.

Figure 1 shows a Bezier curve (p=6), and its basis functions (Bernstein basis functions). It can be noticed that the first basis function, R0, has a significant effect on the start point of the curve, R6 controls the end point of the curve and the remainder of basis functions have no effect on the start and end points. This is one advantage of Bezier curves, and this property reduces the computational time during trajectory optimization. The trajectory shape will vary with variation of the coefficients Ci. The control point that are used in Bezier curve shaping in Figure 1 are C{(1,2),(2,3),(3,2),(4,4),(5,4),(6,0),(7,3)}.

B. Trajectory Profiles Description
The speed profiles in forward (u), lateral (v), and vertical (w) axes can be written by using polynomial functions (6th order Bezier function):
u(τ)=cu0R0(τ)+cu1R1(τ)+⋯+cu6R6(τ)v(τ)=cv0R0(τ)+cv1R1(τ)+⋯+cv6R6(τ)w(τ)=cw0R0(τ)+cw1R1(τ)+⋯+cw6R6(τ)(19)
View SourceRight-click on figure for MathML and additional features.Using 6th order polynomial functions to describe the speed profiles gives a good flexibility over the design horizon with acceptable number design variables (the polynomial coefficients) [15]. Calculation of acceleration, jerk, and position profiles can be done by taking the first derivative of (19) for the acceleration profiles, the second derivative of (19) for the jerk profiles, and integration for the position profiles. In order to do so, a relationship between the curve parameter τ and time t must be defined. A fixed time horizon (th) is used so t can be represented by t=th.τ. Hence the acceleration profiles can be calculated:
u˙(τ)=1th(cu0dR0(τ)dτ+⋯+cu6dR6(τ)dτ)(20)
View SourceRight-click on figure for MathML and additional features.and
d2udt2=1t2h⋅d2udτ2(21)
View SourceRight-click on figure for MathML and additional features.The acceleration and jerk profiles for lateral and vertical axis can be calculated in a similar way. The position profile is driven by integration of the basis function with respect to time t. This can be done by substituting τ=t/th in (18) and then integrating the basis functions with respect to time:
Rinti=∫th0Ri(t)dt;i=0,1,…,6(22)
View SourceRight-click on figure for MathML and additional features.

The receding horizon trajectory profiles are discretized into n steps within the period 0≤τ≤1 to evaluate the cost function at each step during the optimization process. Discretized trajectory profiles can be calculated by discretizing the basis functions into n steps, so the resulted discrete basis functions can be written as matrices as follow:
R=⎛⎝⎜⎜R0(τ1)⋮R6(τ1)⋯⋱⋯R0(τn)⋮R6(τn)⎞⎠⎟⎟(23)
View SourceRight-click on figure for MathML and additional features.

The same procedure can be applied to calculate R′,R′′, and Rint, all these matrices can be calculated off-line, hence the on-line trajectory profiles calculation is reduced to simple matrix multiplication:
u=CuTR,u˙=1thCuTR′,u¨=1t2hCuTR′′,x=x0+CuTRint(24)
View SourceRight-click on figure for MathML and additional features.where CuT is the vector of coefficients for forward axis:
CuT=[cu0  cu1  ⋯  cu6](25)
View SourceRight-click on figure for MathML and additional features.The trajectory profiles for the lateral and vertical axes can be similarly calculated.

By parameterizing the output profiles by the Bezier functions, the optimal control problem is converted into an optimization problem with the design variables being the polynomial coefficients. Hence there are 21 coefficients to be determined (seven for each axis).

C. Initial Conditions
The current aircraft state can be measured by a sensing unit and used as the initial boundary conditions that guarantee a smooth transition from the current state to the target state. Substituting τ=0 in the trajectory profile equations (19), (20), and (21) gives:
cu0=u0,  cu1=th6u˙0+cu0,cu2=t2h30u¨0−cu0+2cu1(26)
View SourceRight-click on figure for MathML and additional features.where u0 is the initial forward speed, u˙0 is the initial forward acceleration and u¨0 is the initial forward jerk. Thus the first three coefficients for each trajectory profile can be determined and the number of design variables reduced from 21 to 12. In order to reduce the computational time of the optimization problem, the aircraft and the obstacles constraints have been augmented in the cost function by using a penalty function method.

D. Aircraft Constraints
In order to ensure that the resulted optimal trajectory will be achieved without exceeding the aircraft performance and control limits (i.e. ensure U∈U,X∈X), the cost function is augmented with additional penalty function terms. The Yukawa potential function [16] is used:
Cp=Ae−αpdpdp(27)
View SourceRight-click on figure for MathML and additional features.where Cp is the aircraft performance constraint term added to the total cost function, Ap is the scaling factor, αp is the decay rate and dp is the performance margin given by:
dp(%)=100−100(current  state  valuestate  max/min  value)(28)
View SourceRight-click on figure for MathML and additional features.To avoid a zero value of dp, a minimum performance margin value dmin must be defined so that:
if  dp≤dmin  then  dp=dmin
View SourceRight-click on figure for MathML and additional features.

It can be clearly seen from (27) that when the performance margin decreases (i.e. the current state value is close to its limit), the potential function takes a huge value. Thus the total cost function will increase significantly, so the search algorithm tries to find another solution that keeps the aircraft state away from its limits.

E. Obstacle Constraints
The collision avoidance constraint, Y∈Y, can be achieved by either including constraints on the optimization process or by augmenting the cost function with a penalty function. The latter is used here so that the total computation time of the optimization process is reduced. As for the performance constraints, the Yukawa potential function is used to punish the cost function if the aircraft approaches an obstacle:
Cob=Aobe−αobdobdob(29)
View SourceRight-click on figure for MathML and additional features.where Cob is a penalty term that represents the obstacle constraints, Aob is a scaling factor, αob is the decay rate and d08 is the distance between the nearest point on the obstacle and the point of interest.

Although using potential functions to describe the obstacle constraints complicates the cost function, it simplifies the search algorithm in the optimization process. Another advantage of using a potential function is that it handles the collision event in a manner which is closer to human behaviour. For example, avoidance manoeuvres can vary according to many factors such as aircraft speed, obstacle speed, aircraft manoeuvrability, and obstacle manoeuvrability. Additionally, due to the difficulty in generating a full 3D illustration for the obstacles that are detected by the on-board sensor unit the potential function approach does not need a 3D description of an obstacle, it just needs the distance between the aircraft and the nearest point in the obstacle [17].

F. Total Cost Function
The following cost function is thus used for the optimization process:
J=∑i=1n[λpJpi+λsJsi+λprfJprfi+λobJobi]+λtJt(30)
View SourceRight-click on figure for MathML and additional features.where

Jpi	
is the position cost function:
Jpi=(xdi−xai)2+(ydi−yai)2+(zdi−zai)2(31)
View SourceRight-click on figure for MathML and additional features.

Jsi	
is the speed cost function:
Jsi=(udi−uai)2+(vdi−vai)2+(wdi−wai)2(32)
View SourceRight-click on figure for MathML and additional features.

Jprfi	
is the vehicle constraints penalty function:
Jprfi=∑j=1qApe−αpdpdp(33)
View SourceRight-click on figure for MathML and additional features.

Jobi	
is the vehicle constraints penalty function:
Jobi=∑j=1mAobe−αobdobdob(34)
View SourceRight-click on figure for MathML and additional features.and
Jt=λh(ψdn−ψan)2+λf(γdn−γan)2(35)
View SourceRight-click on figure for MathML and additional features.where λ are scaling factors, n is the number of points that will be evaluated across the design horizon, q is the number of performance constraints, m is the number of detected obstacles, ψ is the heading angle and γ is the flight path angle. The superscript a means the actual value, while the superscript d means the demanded value.

It can be seen that the cost function given by (30) provides a balance between the different terms; trajectory tracking terms (Jp,Js,Jt) and constraints terms (obstacle avoidance term Job, performance constraint term Jprf). This balance can be controlled by changing the scaling factors λ. The scaling factors can be constants or they may vary according to the situation, in other words the priority of the cost function terms can be varied in order to allow the aircraft to fly safely in different flight scenarios. By augmenting the constraints in the cost function the optimal problem will be solved as an unconstrained optimal problem, thus the computational time will reduced significantly.

G. Avoiding Local Minima
Using a gradient-based method to solve the optimal problem introduces the local minimum problem. The performance constraints tend to act as an enclosing boundary around the entire search space, hence are less likely to result in local minimum. Thus, the obstacle constraints are the primary source of the local minima. When obstacles are detected this can have the impact of dividing the feasible design space into unconnected regions, therefore reducing the effectiveness of the solver of the optimal problem [17]. The possibility of getting trapped in local minimum is reduced by providing a mechanism for the search to jump to the different regions of the design space. This is achieved by generating a set of candidate trajectories then comparing the cost for each candidate then select the one that gives the minimum cost to initiate the optimal problem solver. The candidate trajectories are generated by applying maximum/minimum inputs to the vehicle model with the current vehicle states as initial states to ensure that the maximum performance manoeuvres in each axis are always available if required. In this case the input commands are:
ϕ=[ϕmin  ϕc  ϕmax]T=[Tmin  Tc  Tmax]n=[nmin  nc  nmax](36)
View SourceRight-click on figure for MathML and additional features.where ϕc,Tc, and nc are the current values of the inputs, and ϕmin/max,Tmin/max, and nmin/max are the minimum and maximum values of the inputs which can be calculated from the vehicle specifications (the Aerosonde UAV [?] model and specifications are used here). This combination will produce 33=27− candidate trajectories.

Fig. 2. - System block diagram
Fig. 2.
System block diagram

Show All

SECTION V.Simulation Results
This section demonstrates the method's effectiveness by showing simulation results of different scenarios. Figure 2 shows the system block diagram that is used in MATLAB/Simulink to produce the simulation results. For all scenarios, the global trajectory is level flight with constant speed v=30m.s−1 at 1000 m height, heading ψ=0, the receding horizon time is th=100 s and sampling time ts=0.2 s, the optimization process is updated every 10 seconds. The obstacle is represented as a sphere, and a 4D model of the moving obstacle is generated using a straight projection method [18], which assumes that the obstacle does not manoeuvre during the receding horizon time.

A. Trajectory Tracking and Pop-Up Obstacle Avoidance
In this scenario the initial position of the UAV is higher than the global trajectory but with the same speed and direction. There is also a pop-up obstacle that the UAV must avoid. Figure 3 shows the simulation result of this scenario, it can be seen that the UAV is converging to the global trajectory then when the static obstacle appeared in its way, the UAV performed the necessary manoeuvre in order to avoid the obstacle. Then the UAV converged again to the global trajectory after passing the static obstacle. Figure 4 shows time histories of some state variables of the UAV (position, speed, heading angle ψ, and flight path angle γ) during this scenario.

Fig. 3. - Converging to the global trajectory and avoiding a pop-up obstacle
Fig. 3.
Converging to the global trajectory and avoiding a pop-up obstacle

Show All

Fig. 4. - Position, speed, heading, and flight path during the manoeuvre
Fig. 4.
Position, speed, heading, and flight path during the manoeuvre

Show All

B. Global Trajectory Tracking with Two Moving Intruders
In this case the aircraft encounters two types of intruders so that there are two potential collisions, head-on and overtaking. The UAV has the following initial flight state (level flight at the initial position (0,10,1000) m heading ψ=0, constant speed v=30m.s−1). The first intruder (Intruderl) has the following state (level flight at initial position (2000,10,1000) m, heading ψ=π rad, constant speed v=18m.s−1). The second intruder (Intruder2) has the following initial state (level flight at initial position (2100,10,1000), heading ψ=0 rad, constant speed v=15m.s−1). So Intruderl causes a head-on collision scenario, and then the UAV overtakes Intruder2. The protection zone around each intruder was chosen to be 200 m, so the distance between the UAV and the intruders should not become less than 200 m. Figure 5 shows the UAV trajectory during these scenarios. It can be seen that the UAV avoided both collision scenarios and returned to the global trajectory when it finished overtaking Intruder2. To clarify the performed manoeuvres, the projection of the UAV position on the horizonal and the vertical planes are included in Figure 5. The spheres that appear in Figure 5 represent the protection zones around the intruders when they and the UAV have the same position on the x-axis. Figure 6 gives the time histories of some UAV state variables (position, speed, heading angle ψ, and flight path angle γ) during these scenarios, and it also shows the position state of the intruders. The top-left subplot in Figure 6 shows the x distance time histories of the UAV (solid line), Intruder1 (dashed line), and Intruder2 (dotted line), while the other two subplots in the left column show the y and z distance time histories. It can be noted that when the UAV and one of the intruders have the same x distance, y and z will be at their maximum values, so the UAV is avoiding a conflict with the intruders.

Fig. 5. - Collision avoidance scenarios, head-on (intruderl), overtaking (intruder2)
Fig. 5.
Collision avoidance scenarios, head-on (intruderl), overtaking (intruder2)

Show All

Fig. 6. - Position, speed, heading, and flight path during the manoeuvre
Fig. 6.
Position, speed, heading, and flight path during the manoeuvre

Show All

SECTION VI.Conclusion
An optimal local trajectory generation by using B-spline is proposed for a real-time collision avoidance algorithm. Online avoidance maneuver generation, optimisation, and global trajectory tracking for different conflict scenarios are tested successfully in simulation environment (MATLAB/Simulink).

Although the optimisation solver could be trapped in the local minima due to the obstacles existing, the coarse grid approach that is proposed in IV-G allows the solver to escape the local minima and ensure sufficient coverage of the overall design space. A computational time for the real-time collision avoidance algorithm is reduced significantly by using output space to formulate the optimal problem, and augmenting the vehicle/obstacle constraints in the cost function. The simulation results show that the proposed approach allows the UAV to track a predefined global trajectory as well as avoiding collisions with different types of conflict scenarios in real-time.


Authors

Figures

References

Citations

Keywords

Metrics
More Like This
Differential Flatness-Based Trajectory Planning and Tracking for Fixed-Wing Aircraft in Clustered Environments
2023 42nd Chinese Control Conference (CCC)

Published: 2023

Trajectory Optimization of Aircraft Based on Intelligent Bionic Algorithm
2024 IEEE 7th International Conference on Automation, Electronics and Electrical Engineering (AUTEEE)

Published: 2024

Show More
References

References is not available for this document.
IEEE Personal Account
Change username/password
Purchase Details
Payment Options
View Purchased Documents
Profile Information
Communications Preferences
Profession and Education
Technical interests
Need Help?
US & Canada: +1 800 678 4333
Worldwide: +1 732 981 0060
Contact & Support
Follow
About IEEE Xplore | Contact Us | Help | Accessibility | Terms of Use | Nondiscrimination Policy | IEEE Ethics Reporting | Sitemap | IEEE Privacy Policy

A public charity, IEEE is the world's largest technical professional organization dedicated to advancing technology for the benefit of humanity.

© Copyright 2025 IEEE - All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies.

