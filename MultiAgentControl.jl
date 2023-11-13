using Plots
using LinearAlgebra

# Agent is the supertype for leader and follower both
abstract type Agent end

# Model is the place where the agents operate. Better name would be environment
mutable struct Model1{T<:Agent} 
	t0::Float64
	tf::Float64
	time::Float64
	current_iter::Int64
	dt::Float64
	N::Int64
	leaders::Vector{T}
	followers::Vector{T}
end


function Model1{T}(t0, tf, dt) where T
	current_iter = 1
	time = t0
	N = length(t0:dt:tf)
	return Model1{T}(t0, tf, time, current_iter, dt, N, T[], T[])
end

Model = Model1

mutable struct KP4{T <: Agent} <: Agent
    # Meta states
    state::Vector{Float64}    # Current state of the leader
    selfid::Int64# Id of the agent. If negative the agent is a leader.
    depth::Int64# The depth at which the agent is located
    group_leader_id::Int64  # The id of the grup to which the agent belongs
	name::String

    # Storing the values
    state_hist::Matrix{Float64}# History of states of leader agent
    input_hist::Matrix{Float64}# History of inputs applied to the leader agent

    # Additional information
    neighborsf::Vector{T} # Neighbors of the agent
    neighborsl::Vector{T} # Neighbors of the agent
    neighbors_dist_f::Vector{Float64} # The distance of the neighbors form the leader agent
    neighbors_dist_l::Vector{Float64} # The distance of the neighbors form the leader agent
    local_leader::T # id of the local leader that is to be followed
    local_leader_dist::Float64# id of the local leader that is to be followed

    # Parameters
    communication_radius::Float64
    v::Float64

    # x0 : Initial state of the leader
    # input_dim : Dimension of the input to the system
    # m : Top model in which the system is located
    function KP4{T}(x0, input_dim, v, label::String, name::String, m::Model) where T<:Agent
        A = new{T}()

        sys_dim = length(x0)

        A.state = x0
        A.selfid = length(m.leaders) + length(m.followers) + 1
        A.depth = label == "leader" ? 0 : typemax(Int64)
        A.group_leader_id = label == "leader" ? length(m.leaders) + 1 : 0
		A.name = name

        A.state_hist = zeros(sys_dim, m.N)
        A.state_hist[:, 1] = x0
        A.input_hist = zeros(input_dim, m.N)

        A.neighborsf = T[]
        A.neighborsl = T[]
        A.neighbors_dist_f = Int64[]
        A.neighbors_dist_l = Int64[]
        A.local_leader = A
        A.local_leader_dist = -1     # Id of the neighbor selected for following

        A.communication_radius = 1.5
        A.v = v

        if label == "leader"
            push!(m.leaders, A)
        else
            push!(m.followers, A)
        end

        return A
    end
end

KPf = KP4{KP4}

function compute_dist(ag1::KPf, ag2::KPf)
    return norm(ag1.state - ag2.state)
end

function compute_neighbors_l(ag::KPf)
    empty!(ag.neighborsl)
    empty!(ag.neighbors_dist_l)
    for agent in m.leaders::Vector{KPf}
        if agent != ag
            dist = compute_dist(ag, agent)
            if dist < ag.communication_radius
                push!(ag.neighborsl, agent)
                push!(ag.neighbors_dist_l, dist)
            end
        end
    end
end

function compute_neighbors_f(ag::KPf)
    empty!(ag.neighborsf)
    empty!(ag.neighbors_dist_f)
    for agent in m.followers
        if agent != ag
            dist = compute_dist(ag, agent)
            if dist <= ag.communication_radius
                push!(ag.neighborsf, agent)
                push!(ag.neighbors_dist_f, dist)
            end
        end
    end
end

function compute_local_leader(ag::Agent)
    if isempty(ag.neighborsl)
        for agent in ag.neighborsf
            dist = compute_dist(ag, agent)
            if agent.depth < ag.depth - 1
                ag.local_leader = agent
                ag.local_leader_dist = dist
                ag.depth = agent.depth + 1
                ag.group_leader_id = agent.group_leader_id
            end
        end
    else
        min_dist = Inf
        for agent in ag.neighborsl
            dist = compute_dist(ag, agent)
            if dist < min_dist
                min_dist = dist
                ag.local_leader = agent
                ag.local_leader_dist = dist
                ag.depth = 1
            end
        end
    end
end


function input(ag::Agent, m::Model)
    if ag.depth == 0
		if ag.name == "L1"
        	u = -3π/4
		elseif ag.name == "L2"
        	u = π/4
		elseif ag.name == "L3"
        	u = π/2
		end
    else
        xn = ag.local_leader.state
        x = ag.state
        u = atan(xn[2] - x[2], xn[1] - x[1])
        # println(u)
        # u = 0 
    end
end

function dynamics(ag::KPf, x, u, t)
    return [ag.v * cos(u), ag.v * sin(u)]
end

function step!(ag::Agent, input::Function, m::Model; method=:euler)
    x = ag.state
    u = input(ag, m)
    t = m.time

    ag.state .+= dynamics(ag, x, u, t) * m.dt # x, u, t kept in order to implement advanced stepping

    ag.state_hist[:, m.current_iter+1] = x
    ag.input_hist[m.current_iter+1] = u
end

t0 = 0.0
tf = 10.0
dt = 0.1


m = Model{KPf}(t0, tf, dt)

L1 = KPf([0.0, 0.0], 1, 0.2, "leader", "L1", m)
L2 = KPf([3.0, 2.5], 1, 0.2, "leader", "L2", m)
L3 = KPf([-1.0, 5.0], 1, 0.2, "leader", "L3", m)
A1 = KPf([1.0, 0.0], 1, 0.6, "follower", "A1", m)
A2 = KPf([1.0, 1.0], 1, 0.6, "follower", "A2", m)
A3 = KPf([0.0, 1.0], 1, 0.6, "follower", "A3", m)
A4 = KPf([-1.0, 2.0], 1, 0.6, "follower", "A4", m) 
A5 = KPf([2.0, 0.3], 1, 0.6, "follower", "A5", m)
A6 = KPf([1.2, 1.8], 1, 0.6, "follower","A6", m)
A7 = KPf([2.0, 1.4], 1, 0.6, "follower", "A7", m)
A8 = KPf([0.3, 1.8], 1, 0.6, "follower", "A8", m)
A9 = KPf([1.6, 2.6], 1, 0.6, "follower", "A9", m)
A10 = KPf([3.0, 1.0], 1, 0.6, "follower", "A10", m)
A11 = KPf([0.6, 2.7], 1, 0.6, "follower", "A11", m)
A12 = KPf([0.3, 3.7], 1, 0.6, "follower", "A12", m)
A13 = KPf([-0.7, 3.0], 1, 0.6, "follower", "A13", m)
A14 = KPf([-1.0, 4.0], 1, 0.6, "follower", "A14", m)
A15 = KPf([-2.2, 3.3], 1, 0.6, "follower", "A15", m)
A16 = KPf([0.2, 4.6], 1, 0.6, "follower", "A16", m)
A17 = KPf([0.85, 5.0], 1, 0.6, "follower", "A17", m)
A18 = KPf([-2.2, 4.21], 1, 0.6, "follower", "A18", m)
A19 = KPf([1.6, 4.0], 1, 0.6, "follower", "A19", m)
A20 = KPf([-2.3, 2.1], 1, 0.6, "follower", "A20", m)

for agent in [m.leaders; m.followers]
    compute_neighbors_l(agent)
    compute_neighbors_f(agent)
end

for _ in [m.leaders; m.followers]
    for agent in [m.leaders; m.followers]
        if agent.depth != 0      # Depth is 0 for leader agents
            compute_local_leader(agent)
        end
    end
end

println("EOC")

for i = 1:m.N-1
    for agent in [m.leaders; m.followers]
        step!(agent, input, m)
    end
    m.current_iter += 1
end


fig1 = plot(aspect_ratio=1, legend= nothing)
for l in m.leaders
	plot!(fig1, l.state_hist[1, :], l.state_hist[2, :])
end
for f in m.followers
	plot!(fig1, f.state_hist[1, :], f.state_hist[2, :])
end
display(fig1)