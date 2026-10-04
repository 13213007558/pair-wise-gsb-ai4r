# frozen_string_literal: true

# A long deterministic corridor: the agent observes the reward only once, at
# the terminal end. Prioritized sweeping then propagates that reward back
# through the previously observed predecessors, one planning step at a time.

require 'ai4r/reinforcement/prioritized_sweeping_agent'

agent = Ai4r::Reinforcement::PrioritizedSweepingAgent.new(rng: Random.new(7))
agent.set_parameters(learning_rate: 1.0, discount: 1.0, exploration: 0.0)

states = (0..6).map { |i| :"s#{i}" }
states.each { |state| agent.declare_available_actions(state, [:right]) }
agent.declare_available_actions(:goal, [])

# Experience phase: walk the corridor once, doing no planning yet.
states.each_cons(2) do |state, next_state|
  agent.update(state, :right, 0, next_state, planning_budget: 0)
end
agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)

puts "interaction updates: #{agent.interaction_updates}"
puts "planning updates:    #{agent.planning_updates}"
puts "Q(s0, right) before planning: #{agent.q[:s0][:right]}"
puts

# Planning phase: each call returns the state-actions swept, highest residual
# first. Notice the order runs backwards from the reward.
[3, 4].each do |budget|
  swept = agent.plan(budget)
  puts "plan(#{budget}) swept: #{swept.map { |state, action| "#{state}->#{action}" }.join(', ')}"
  puts "Q(s0, right) = #{agent.q[:s0][:right]}, pending: #{agent.pending_queue_size}"
end
