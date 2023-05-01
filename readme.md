#readme

1. Each crowd worker has the concept of an account, with Humanity and Staking values
	The voting power for generating the social graph is the sum of the Humanity Score and the square root of the Staking amount

2. Receive a list of addresses to be analyzed and randomly select 9 of them, and the crowd worker selects 3 of them
 This is repeated for each crowdworker

3. Finally, the reward is calculated based on the correlation of the answers given by each crowdworker. At this point, based on the correlation of the responses and the voting power, the contribution is calculated to simulate how much of a percentage of the reward will be received in a single analysis session.

4. The addresses to be analyzed are classified into several clusters.