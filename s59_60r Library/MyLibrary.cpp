













































//############################################################//
//               1.  Sum of two Big Numbers                   //

string findSum(string str1, string str2) {

	if (str1.length() > str2.length()) swap(str1, str2);

	string str = "";

	int n1 = str1.length(), n2 = str2.length();

	reverse(str1.begin(), str1.end());
	reverse(str2.begin(), str2.end());

	int carry = 0;
	for (int i = 0; i < n1; i++) {
		int sum = ( ( str1[i] - '0' ) + ( str2[i] - '0' ) + carry );
		str.push_back(sum % 10 + '0');
		carry = sum / 10;
	}

	for (int i = n1; i < n2; i++) {
		int sum = ((str2[i] - '0') + carry);
		str.push_back(sum % 10 + '0');
		carry = sum / 10;
	}
	if (carry) str.push_back(carry + '0');
	reverse(str.begin(), str.end());
	return str;
}





























//############################################################//
//                       2.  			t Trees                   //

template < typename T >
class SegmentTree {
private:
	T *Stree;
	T *Snum;
	int n;
	T NullValue = 0;
	T combine(T a, T b) {
		return a + b;

	}
	void build(int idx, int s, int e) {
		if (s > e)
			return;
		if (s == e) {
			Stree[idx] = Snum[s];
			return ;
		}
		int m = ( s + e ) / 2;
		build( 2 * idx , s , m);
		build( 2 * idx + 1 , m + 1 , e);
		Stree[idx] = combine(Stree[2 * idx] , Stree[2 * idx + 1] );
		return;
	}
	T queryUtil(int idx, int s, int e, int l, int r) {
		if (l > e || r < s)
			return NullValue;
		if (s >= l && e <= r) {
			return Stree[idx];
		}
		int m = ( s + e ) / 2;
		T left = queryUtil(2 * idx , s , m , l , r);
		T right = queryUtil(2 * idx + 1 , m + 1 , e , l , r);
		return combine(left , right );
	}
	void updateUtil(int idx, int s, int e, int l, int r, int val) {
		if (r < s || l > e)
			return;
		if (s == e) {
			Stree[idx] = val;
			return;
		}
		int m = ( s + e ) / 2;
		updateUtil(2 * idx , s , m , l , r , val);
		updateUtil(2 * idx + 1 , m + 1 , e , l  , r , val);
		Stree[idx] = combine( Stree[2 * idx] , Stree[2 * idx + 1]);
		return;
	}
public:
	SegmentTree(vector<T> a) {
		this->n = a.size();
		Stree = new T[4 * n + 5];
		Snum = new T[n];

		for (int i = 0; i < n; i++) {
			Snum[i] = a[i];
		}

		build(1, 0, n - 1);
	}
	// 1-based
	T query(int l, int r) {
		return queryUtil(1, 0, n - 1, l - 1, r - 1);
	}

	void update(int x, int val) {
		updateUtil(1, 0, n - 1, x - 1, x - 1, val);
	}
};














































//############################################################//
//        3.  Segment Trees with lazy propagation            //







//#########################################################//
//                 4.Binary Exponentiation                 //


int power(int a, int b, int mod = LONG_MAX) {
	int res = 1;
	while (b > 0) {
		if (b & 1)
			res = (res * a) % mod;
		a = (a * a) % mod;
		b >>= 1;
	}
	return res;
}
















































//#########################################################//
//                 5.Divisors of Number                   //

vector<int> divisors(int n) {
	vector<int> res;
	for (int i = 1; i * i <= n; i++) {
		int one = i;
		if (n % one == 0) {
			res.push_back(one);
			if (one != n / one)
				res.push_back(n / one);
		}
	}
	return res;
}















































//#########################################################//
//                 6.Maximum Sum Subarray                  //

int maxSubarraySum(vector<int> a) {
	int maxi = a[0];
	int cur = a[0];
	for (int i = 1; i < a.size(); i++) {
		cur = max(a[i], cur + a[i]);
		maxi = max(cur, maxi);
	}
	return maxi;
}












































//#########################################################//
//                 7.Check if a Number is Prime            //



bool isPrime(int n) {
	if (n == 1)return false;
	if (n == 2)return true;
	for (int i = 2; i * i <= n; i++)
		if (n % i == 0)return false;
	return true;
}










































//#########################################################//
//                   8.Sort Three Numbers                  //

void Tsort(int &a, int &b, int &c) {
	if (b < a) swap(a, b);
	if (c < b) { swap(b, c); if (b < a) swap(b, a);}
	return;
}










































//#########################################################//
//                 9. Prime Factors of Number                   //

map<int, int> primeFactors(int n) {
	map<int, int> res;
	while (n % 2 == 0) {res[2]++; n = n / 2;}
	for (int i = 3; i * i <= n; i += 2)while (n % i == 0) {res[i]++; n = n / i;}
	if (n > 2)res[n]++;
	return res;
}
















































//##################################################################//
//                 10. Prime Numbers less than N                    //


vector<int> SieveOfEratosthenes(int n) {

	bool prime[n + 1];
	memset(prime, true, sizeof(prime));
	vector<int> res;
	for (int p = 2; p * p <= n; p++) {
		if (prime[p] == true) {
			for (int i = p * p; i <= n; i += p)
				prime[i] = false;
		}
	}

	for (int p = 2; p <= n; p++)
		if (prime[p])
			res.push_back( p);
	return res;
}














































//##################################################################//
//                 11. Removing Rightmost Set Bit                  //

int remove_rightmost_bit(int n) {
	return n - (n & (-n));
}















































//##################################################################//
//                     12. Fenwick Tree                             //


class FenwickTree {
private:
	int n;
	int *Tree;
public:
	void update(int i, int inc, int n) {
		while (i <= n) {
			Tree[i] += inc;
			i += (i & (-i));
		}
	}
	void build(vector<int> &a) {
		for (int i = 1; i <= a.size(); i++) {
			update(i, a[i - 1], a.size());
		}
		return ;
	}
	int query(int i) {
		int res = 0;
		while (i > 0) {
			res += (Tree[i]);
			i -= (i & (-i));
		}
		return res;
	}
	FenwickTree(vector<int> a) {
		n = a.size();
		Tree = new int[2 * n];
		build(a);
	}

	int queryRange(int l, int r) {
		return query(r) - query(l - 1);
	}
};
















































//##################################################################//
//                     12. nCr Mod P                             //



// Slow But Clean
int nCrModp(int n, int r, int mod = LONG_MAX) {
	if (r > n - r) r = n - r;

	int C[r + 1];
	memset(C, 0, sizeof(C));

	C[0] = 1;
	for (int i = 1; i <= n; i++) {
		for (int j = min(i, r); j > 0; j--)
			C[j] = (C[j] + C[j - 1]) % mod;
	}

	return C[r];
}


// Faster but maybe faulty

int fac[N];
fac[0] = 1;
for (int i = 1; i <N; i++)
	fac[i] = (fac[i - 1] * i) % MOD;

int power(int a, int b, int mod) {
	int res = 1;
	while (b > 0) {
		if (b & 1)
			res = (res * a) % mod;
		a = (a * a) % mod;
		b >>= 1;
	}
	return res;
}
int modInverse(int n, int p) {
	return power(n, p - 2, p);
}

int nCrModPFermat(int n, int r, int p) {
	if (r == 0)
		return 1;
	return (fac[n] * modInverse(fac[r], p) % p * modInverse(fac[n - r], p) % p) % p;
}


// Runs in O(r)


int nCrOT(int n, int k,int mod){
    int res = 1;
    if (k > n - k) k = n - k;
    function<int (int, int, int)> modDivide = [&](int a,int b,int m) ->decltype(a){
    	function<int (int, int, int& ,int &)> gce = [&](int aa, int bb, int &x, int &y){
		    if (aa == 0){
		        x = 0, y = 1;
		        return bb;
		    }
		    int x1, y1,gcd = gce(bb%aa, aa, x1, y1);
		    x = y1 - (bb/aa) * x1;
		    y = x1;
		    return gcd;
		} ;
		function<int (int, int)> modInverse = [&](int bb,int mm) ->decltype(bb){
			int x, y;
		    int g = gce(bb, mm, x, y);
		    if (g != 1)
		        return -1;
		    int res = (x%mm + mm) % mm;
		    return res;
		};

	    a = a % m;
	    int inv = modInverse(b, m);
	    if (inv == -1)
	       assert(0);
	    return  (inv * a) % m;
    };

    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res %= mod;
        res = modDivide(res,(i+1),mod);
        res %= mod;
    }
    return res;
}






































//##################################################################//
//                           14. Trie                              //


class Trie {
private:
	class Node {
	public:
		char data;
		bool isTerminal;
		int cnt = 0;
		map<char, Node*> children;

		Node(char d) {
			this->data = d;
			this->isTerminal = false;
			this->cnt = 0;
		}
	};
public:
	Node *root;

	Trie() {
		this->root = new Node('\0');
	}

	void addWord(string s) {
		Node *temp = root;
		for (int i = 0; i < s.length(); i++) {
			if (temp->children.count(s[i])) {
				temp = temp->children[s[i]];
			} else {
				Node *n = new Node(s[i]);
				temp ->children[s[i]] = n;
				temp = n;
			}
			temp->cnt++;
		}
		temp->isTerminal = true;
	}

	bool searchWord(string s) {
		Node *temp = root;
		for (int i = 0; i < s.length(); i++) {
			if (temp->children.count(s[i]) and temp->children[s[i]]->cnt > 0) {
				temp = temp->children[s[i]];
			} else {
				return false;
			}
		}
		return temp->isTerminal;
	}
	void removeWord(string s) {
		Node *temp = root;
		for (int i = 0; i < s.length(); i++) {
			if (temp->children.count(s[i])) {
				temp = temp->children[s[i]];
			} else {
				Node *n = new Node(s[i]);
				temp ->children[s[i]] = n;
				temp = n;
			}
			temp->cnt--;
		}
		temp->isTerminal = true;
	}
};















































//##################################################################//
//                  15. Check for valid triangle                   //


bool isValidTriangle(int a, int b, int c) {
	if (a + b <= c || a + c <= b || b + c <= a)
		return false;
	else
		return true;
}















































//##################################################################//
//                16. Connected Components of Graphs                 //



//--------------------    Nodes are numbers  -----------------------//
void dfs(int src, map<int, set<int>> &adj, map<int, bool> &visit,set<int> &nodes,set<int> &edges) {
	visit[src] = true;
	nodes.insert(src);

	for (int x : adj[src]) {
		if (!visit[x]) {
			dfs(x, adj, visit,nodes,edges);
		}
		if(x!=src){
			edges.insert({src,x});
			edges.insert({x,src});
		}
	}
}
int ConnectedComponent(map<int, set<int>> &adj, int n) {
	map<int, bool> visit;
	int res = 0;
	for (int i = 1; i <= n; i++) {
		if (!visit[i]) {
			res++;
			set<int> nodes;
			set<int> edges;
			dfs(i, adj, visit,nodes,edges);
		}
	}

	return res;
}





//##################################################################//
//                   49. Make Grid to Graph Adj                     //


int getNode(int x,int y,int n,int m){
	return m*(x) + y+1;
}

pii getCell(int x,int n,int m){
	int temp = x/m;
	if(x%m!=0){
		return {temp,x%m - 1};
	}else{
		return {temp-1,m-1};
	}
}

set<int> adj[n*m+1];



/// *********** LRUD *************
for(int i = 0 ; i < n; i++) {
	for(int j = 0 ; j < m ; j++) {
		if(grid[i][j]!='.') continue;
		// U-D

		for(int x: {-1,1}) {
			if(i+x>=0 and i+x<n and grid[i+x][j]=='.') {
				adj[getNode(i,j,n,m)].insert(getNode(i+x,j,n,m));
				adj[getNode(i+x,j,n,m)].insert(getNode(i,j,n,m));
			}
		}
		// L-R
		for(int x: {-1,1}) {
			if(j+x>=0 and j+x<m and grid[i][j+x]=='.') {
				adj[getNode(i,j,n,m)].insert(getNode(i,j+x,n,m));
				adj[getNode(i,j+x,n,m)].insert(getNode(i,j,n,m));
			}
		}
	}
}




/// *********** Diag-LRUD *************
for(int i = 0 ; i < n; i++) {
	for(int j = 0 ; j < m ; j++) {
		if(grid[i][j]!='.') continue;

		for(int x: {-1,0,1}) {
			for(int y: {-1,0,1}){
				if(i+x>=0 and i+x<n and j+y>=0 and j+y<m and grid[i+x][j+y]=='.') {
					adj[getNode(i,j,n,m)].insert(getNode(i+x,j+y,n,m));
					adj[getNode(i+x,j+y,n,m)].insert(getNode(i,j,n,m));
				}
			}
		}
	}
}






//--------------------    Nodes are cells  -----------------------//
















//##################################################################//
//              17. Generate all Subsequence and Subarray of Array              //


template < typename T >
vector<T> generateSubsequence(T a) {
	vector<T> res;
	int subs = powl(2, a.size()) - 1;
	for (int i = 1; i <= subs; i++) {
		int mask = i;
		T cur;
		for (int j = 0; j < a.size(); j++) {
			if (mask & 1) {
				cur.push_back(a[j]);
			}
			mask /= 2;
		}
		res.push_back(cur);
	}
	return res;
}

template < typename T >
vector<T> generateSubarray(T &a) {
	vector<T> res;
	for (int i = 0; i <a.size(); i++) {
		for (int j = i; j <a.size(); j++){
			T cur;
			for(int k=i;k<=j;k++)
				cur.push_back(a[k]);

			res.push_back(cur);
		}
	}
	return res;
}






























//##################################################################//
//              18. Rotate Grid 90 Degree Anticlockwise              //

vector<vector<char>> rotate90AntiClockwise(vector<vector<char>> grid) {
	vector< vector<char> > res;
	int n = grid.size();
	int m = grid[0].size();
	for (int i = m - 1; i >= 0; i--) {
		vector<char> temp;
		for (int j = 0; j < n; j++) {
			temp.push_back(grid[j][i]);
		}
		res.push_back(temp);
	}
	return res;
}













































//##################################################################//
//			              18. Generate An Array                    //

vector<int> generateArray(int n) {
	vector<int> res;
	int elem_size = 100;
	for (int i = 0; i < n; i++) {
		res.push_back(rand() % elem_size + 1);
	}
	return res;
}


























//##################################################################//
//			              19. Detect Cycle in Graph                //


void dfs(int src, map<int, bool> &visit, map<int, set<int>> &adj, bool &res, int parent) {
	visit[src] = true;
	for (auto x : adj[src]) {
		if (!visit[x]) {
			dfs(x, visit, adj, res, src);
		} else {
			if (x != parent) {
				res = true;
			}
		}
	}
}
bool hasCycle(map<int, set<int>> &adj, int n) {
	map<int, bool> visit;
	bool res = false;
	for (int i = 1; i <= n; i++) {
		if (!visit[i]) {

			dfs(i, visit, adj, res, -1);
			if (res) {
				return true;
			}
		}
	}
	return res;
}





























//##################################################################//
//			              19. Generate Graph                       //


pair<pair<int, int>, vector<pair<int, int>>> generateGraph(int N) {
	srand(time(NULL));
	int n = rand() % N + 3;
	int m = min((int)rand() % N + 3, (n * (n - 1)) / 2);
	pair<int, int> nm = {n, m};
	vector<pair<int, int>> edges;
	for (int i = 0; i < m; i++) {
		int x = rand() % n + 1;
		int y = rand() % n + 1;
		edges.push_back({x, y});
	}
	return {nm, edges};
}































//##################################################################//
//			              19. Visualize Graph                      //


void graphToVisual() {
	int n, m;
	cin >> n >> m;
	int mat[n][n];
	memset(mat, 0, sizeof mat);
	for (int i = 0; i < m; i++) {
		int x, y;
		cin >> x >> y;
		mat[x - 1][y - 1] = 1;
		mat[y - 1][x - 1] = 1;
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << mat[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	for (int i = 0; i < n; i++) {
		cout << i + 1 << '\n';
	}
}







































//##################################################################//
//			            20. Depth First Search                     //


void dfs(int src, map<int, set<int>> &adj, map<int, int> &visit) {
	visit[src] = true;
	cout << src << ' ';
	for (int x : adj[src]) {
		if (!visit[x]) {
			dfs(x, adj, visit);
		}
	}
}









































//##################################################################//
//			            21. Breadth First Search                       //


void bfs(int src, map<int, set<int>> &adj) {

	queue<int> que;
	map<int, int> visit;

	que.push(src);
	visit[src] = true;

	while (!que.empty()) {
		int current_node = que.front();
		cout << current_node << ' ';

		que.pop_front();

		for (int x : adj[current_node]) {
			if (!visit[x]) {
				que.push(x);
				visit[x] = true;
			}
		}

	}

}
































//##################################################################//
//	       22. Longest Increasing Subarray and Subsequence           //


int LongestIncreasingSubarray(vector<int> &a) {
	int res = 1, cur = 1;
	for (int i = 1; i < a.size(); i++) {
		cur += a[i] > a[i - 1] ? 1 : -cur + 1;
		res = max(res, cur);
	}
	return res;
}

// N^2
int LongestIncreasingSubsequence(vi &a, int n ) {
	int dp[n];
	dp[0] = 1;
	for (int i = 1; i < n; i++ ) {
		dp[i] = 1;
		for (int j = 0; j < i; j++ )
			if ( a[i] > a[j] && dp[i] < dp[j] + 1)
				dp[i] = dp[j] + 1;
	}
	return *max_element(dp, dp + n);
}







































//##################################################################//
//			          22. Round Grid Traverse                      //




void spiralPrint(int n, int m, int a[300][300]) {
	int i, k = 0, l = 0;

	while (k < n && l < m) {
		for (i = l; i < m; ++i) {
			cout << a[k][i] << " ";
		}
		k++;

		for (i = k; i < n; ++i) {
			cout << a[i][m - 1] << " ";
		}
		m--;
		if (k < n) {
			for (i = m - 1; i >= l; --i) {
				cout << a[n - 1][i] << " ";
			}
			n--;
		}
		if (l < m) {
			for (i = n - 1; i >= k; --i) {
				cout << a[i][l] << " ";
			}
			l++;
		}
	}
}








































//##################################################################//
//                    23. Log of n Base r                           //

double Logn(double n, double r) {
	return log2l(n) / log2l(r);
}








































//##################################################################//
//                 24. Ones Compliment of a number                  //

unsigned int onesComplement(unsigned int n) {
	int number_of_bits = floor(log2(n)) + 1;
	return ((1 << number_of_bits) - 1) ^ n;
}




































//##################################################################//
//                        25. Modular Division                   //

int modDivide(int a, int b, int m){
	function<int (int, int, int& ,int &)> gce = [&](int aa, int bb, int &x, int &y){
	    if (aa == 0){
	        x = 0, y = 1;
	        return bb;
	    }
	    int x1, y1,gcd = gce(bb%aa, aa, x1, y1);
	    x = y1 - (bb/aa) * x1;
	    y = x1;
	    return gcd;
	} ;
	function<int (int, int)> modInverse = [&](int bb,int mm) ->decltype(bb){
		int x, y;
	    int g = gce(bb, mm, x, y);
	    if (g != 1)
	        return -1;
	    int res = (x%mm + mm) % mm;
	    return res;
	};

    a = a % m;
    int inv = modInverse(b, m);
    if (inv == -1)
       assert(0);
    return  (inv * a) % m;
}



//##################################################################//
//                            25. Rabin Karp                       //



class RabinKarp {
	int * hashh;
	int * poww;
	const int base = 137;
	const int MOD = 1000000009;


	static int power(int a, int b, int MOD) {
		int res = 1;
		while (b > 0) {
			if (b & 1)
				res = (res * a) % MOD;
			a = (a * a) % MOD;
			b >>= 1;
		}
		return res;
	}

	static int gcdExtended(int a, int b, int *x, int *y) {
		if (a == 0) {
			*x = 0, *y = 1;
			return b;
		}
		int x1, y1;
		int gcd = gcdExtended(b % a, a, &x1, &y1);
		*x = y1 - (b / a) * x1;
		*y = x1;
		return gcd;
	}

	static int modInverse(int b, int m) {
		int x, y;
		int g = gcdExtended(b, m, &x, &y);
		if (g != 1)
			return -1;
		return (x % m + m) % m;
	}

	static int modDivide(int a, int b, int m) {
		a = a % m;
		int inv = modInverse(b, m);
		if (inv == -1)
			return -1;
		else
			return (inv * a) % m;
	}

public:
	RabinKarp(int n) {
		poww = new int[n];

		for (int i = 0; i < n; i++) {
			poww[i] = power(base, i, MOD);
		}
	}

	void fit(string &a) {
		hashh = new int[a.length() + 1];
		hashh[0] = 0;
		hashh[1] = (int)a[0];
		for (int i = 1; i < a.length(); i++) {
			hashh[i + 1] = hashh[i] + a[i] * poww[i];
			hashh[i + 1] %= MOD;
		}

	}

	int getHash(int l, int r) {
		int res = (hashh[r + 1] - hashh[l] + MOD) % MOD;
		res = modDivide(res, poww[l], MOD);
		return res;
	}

};





//##################################################################//
//                     25. Rabin Karp w Double Hash                //

class RabinKarpDoubleHash{
    int * hashh;
    int * hashh2;
    int * poww;
    int * poww2;
    int base = 137;
    int base2 = 37;
    const int mod = MOD;


	static int modDivide(int a, int b, int m){
		function<int (int, int, int& ,int &)> gce = [&](int aa, int bb, int &x, int &y){
		    if (aa == 0){
		        x = 0, y = 1;
		        return bb;
		    }
		    int x1, y1,gcd = gce(bb%aa, aa, x1, y1);
		    x = y1 - (bb/aa) * x1;
		    y = x1;
		    return gcd;
		} ;
		function<int (int, int)> modInverse = [&](int bb,int mm) ->decltype(bb){
			int x, y;
		    int g = gce(bb, mm, x, y);
		    if (g != 1)
		        return -1;
		    int res = (x%mm + mm) % mm;
		    return res;
		};
	    a = a % m;
	    int inv = modInverse(b, m);
	    if (inv == -1)
	       assert(0);
	    return  (inv * a) % m;
	}

public:

    RabinKarp(int n) {
        poww = new int[n];
        poww[0] = 1;
        for (long i = 1; i < n; i++) {
            poww[i] = poww[i-1]*base;
            poww[i]%=mod;
        }

        poww2 = new int[n];
        poww2[0] = 1;
        for (long i = 1; i < n; i++) {
            poww2[i] = poww2[i-1]*base2;
            poww2[i]%=mod;
        }
    }

    void fit(string &a) {
        hashh = new int[a.length() + 1];
        hashh[0] = 0;
        hashh[1] = (int)a[0];
        for (int i = 1; i < a.length(); i++) {
            hashh[i + 1] = hashh[i] + a[i] * poww[i];
            hashh[i + 1] %= mod;
        }
        hashh2 = new int[a.length() + 1];
        hashh2[0] = 0;
        hashh2[1] = (int)a[0];
        for (int i = 1; i < a.length(); i++) {
            hashh2[i + 1] = hashh2[i] + a[i] * poww2[i];
            hashh2[i + 1] %= mod;
        }
    }

 	// 1-Based
    pair<int,int> getHash(int l, int r) {
        l--;
        r--;
        int res = (hashh[r + 1] - hashh[l] + mod) % mod;
        res = modDivide(res, poww[l], mod);
        int res2 = (hashh2[r + 1] - hashh2[l] + mod) % mod;
        res2 = modDivide(res2, poww2[l], mod);
        return {res,res2};
    }
};

























//##################################################################//
//                  26. Zig Zag Matrix Traversal                    //

vector<vector<int>> ZigZagMatrix(int matrix[][505], int ROW, int COL) {
	vector<vector<int>> res;
	for (int line = 1; line <= (ROW + COL - 1); line++) {
		int start_col =  max(0LL, line - ROW);
		int count = min(line, min((COL - start_col), ROW));
		vector<int> temp;
		for (int j = 0; j < count; j++)
			temp.push_back(matrix[min(ROW, line) - j - 1][start_col + j]);
		res.push_back(temp);
	}
	return res;
}






























//##################################################################//
//                27. Find if a two points intersect               //


bool doesIntersect(pair<int, int> pt1, pair<int, int> pt2) {
	int x1 = pt1.first,  x2 = pt1.second,  y1 = pt2.first, y2 = pt2.second;
	return (x1 >= y1 && x1 <= y2) || (x2 >= y1 && x2 <= y2) || (y1 >= x1 && y1 <= x2) || (y2 >= x1 && y2 <= x2);
}


































//##################################################################//
//                28. Shortest Path Between Two Nodes               //

int shortestPath(int u, int v, map<int, set<int>> &adj, int n) {
	vector<int> visit(n + 1, 0), dist(n + 1, 0);

	queue<int> Q;
	Q.push(u);
	while (!Q.empty()) {
		int node = Q.front();
		Q.pop();
		for (int x : adj[node]) {
			if (!visit[x]) {
				dist[x] = dist[node] + 1;
				Q.push(x);
				visit[x] = true;
			}
		}
	}
	if (!visit[v]) return LONG_MAX;
	return dist[v];
}



//##################################################################//
//                    29. Maximum Subarray Queries                  //


class MaximumSubarrayQueries {
private:
	struct node {
		int sum, prefixsum, suffixsum, maxsum;
	};
	int N;
	node *tree;
	int n;
	int *a;

	void build(int arr[], int low, int high, int index) {
		if (low == high) {
			tree[index].sum = arr[low];
			tree[index].prefixsum = arr[low];
			tree[index].suffixsum = arr[low];
			tree[index].maxsum = arr[low];
		}
		else {
			int mid = (low + high) / 2;
			build(arr, low, mid, 2 * index + 1);
			build(arr, mid + 1, high, 2 * index + 2);
			tree[index].sum = tree[2 * index + 1].sum +  tree[2 * index + 2].sum;
			tree[index].prefixsum =  max(tree[2 * index + 1].prefixsum, tree[2 * index + 1].sum +  tree[2 * index + 2].prefixsum);
			tree[index].suffixsum =  max(tree[2 * index + 2].suffixsum, tree[2 * index + 2].sum + tree[2 * index + 1].suffixsum);
			tree[index].maxsum = max(tree[index].prefixsum, max(tree[index].suffixsum, max(tree[2 * index + 1].maxsum, max(tree[2 * index + 2].maxsum, tree[2 * index + 1].suffixsum + tree[2 * index + 2].prefixsum))));
		}
	}


	void update_util(int arr[], int index, int low, int high, int idx, int value) {
		if (low == high) {
			tree[index].sum = value;
			tree[index].prefixsum = value;
			tree[index].suffixsum = value;
			tree[index].maxsum = value;
		} else {
			int mid = (low + high) / 2;
			if (idx <= mid) update_util(arr, 2 * index + 1, low, mid, idx, value);
			else update_util(arr, 2 * index + 2, mid + 1, high, idx, value);
			tree[index].sum = tree[2 * index + 1].sum +  tree[2 * index + 2].sum;
			tree[index].prefixsum = max(tree[2 * index + 1].prefixsum, tree[2 * index + 1].sum +  tree[2 * index + 2].prefixsum);

			tree[index].suffixsum =  max(tree[2 * index + 2].suffixsum, tree[2 * index + 2].sum +  tree[2 * index + 1].suffixsum);
			tree[index].maxsum = max(tree[index].prefixsum, max(tree[index].suffixsum,  max(tree[2 * index + 1].maxsum, max(tree[2 * index + 2].maxsum, tree[2 * index + 1].suffixsum + tree[2 * index + 2].prefixsum))));
		}
	}


	node query_util(int arr[], int index, int low, int high, int l, int r) {

		node result;
		result.sum = result.prefixsum =  result.suffixsum = result.maxsum = (LLONG_MIN / 1000000);
		if (r < low || high < l) return result;
		if (l <= low && high <= r) return tree[index];
		int mid = (low + high) / 2;
		if (l > mid)  return query_util(arr, 2 * index + 2, mid + 1, high, l, r);
		if (r <= mid) return query_util(arr, 2 * index + 1, low, mid, l, r);
		node left = query_util(arr, 2 * index + 1, low, mid, l, r);
		node right = query_util(arr, 2 * index + 2, mid + 1, high, l, r);
		result.sum = left.sum + right.sum;
		result.prefixsum = max(left.prefixsum, left.sum + right.prefixsum);
		result.suffixsum = max(right.suffixsum, right.sum + left.suffixsum);
		result.maxsum = max(result.prefixsum, max(result.suffixsum, max(left.maxsum, max(right.maxsum, left.suffixsum + right.prefixsum))));
		return result;
	}

public:
	MaximumSubarrayQueries(vector<int> b) {
		this->N = b.size();
		this->n = b.size();
		tree = new node[4 * N + 10];
		a = new int[N];

		for (int i = 0; i < N; i++) {
			a[i] = b[i];
		}
		build(a, 0, n - 1, 0);
	}
	int query(int l, int r) {
		return query_util(a, 0, 0, n - 1, l - 1, r - 1).maxsum;
	}
	void update(int x, int value) {
		a[x - 1] = value;
		update_util(a, 0, 0, n - 1, x - 1, value);
		return ;
	}
};









//##################################################################//
//                        30. Trio Class                           //



template < typename A, typename B , typename C>
class trio {
public: A first; B second; C third;
	trio(A a, B b, C c) {this->first = a; this->second = b; this->third = c;}
	trio() {this->first = 0; this->second = 0; this->third = 0;}
};
template < typename A, typename B , typename C>
ostream& operator<<(ostream& os, const trio<A, B, C> &p) {return os << "(" << p.F << ", " << p.S << ", " << p.third << ")";}



















//##################################################################//
//                        31. KMP Preprocess                        //
template < typename T >
vector<T> KMP_Preprocess(vector<T> &s) {
	vector<T> pi(s.size());
	for (int i = 1; i < s.size(); i++) {
		int j = pi[i - 1];
		while (j > 0 && s[i] != s[j])
			j = pi[j - 1];
		if (s[i] == s[j])
			j++;
		pi[i] = j;
	}
	return pi;
}














//##################################################################//
//            32. Find two number with sum S and xor X              //

pair<int, int> compute( int S, int X) {
	int A = (S - X) / 2;

	int a = 0, b = 0;

	for (int i = 0; i <= ceil(log2l(S)); i++) {
		int Xi = (X & (int)powl(2, i));
		int Ai = (A & (int)powl(2, i));
		if (Xi == 0 && Ai == 0) {

		} else if (Xi == 0 && Ai > 0) {
			a = ((int)powl(2, i) | a);
			b = ((int)powl(2, i) | b);
		} else if (Xi > 0 && Ai == 0) {
			a = ((int)powl(2, i) | a);
		} else {
			return { -1, -1};
		}
	}
	return {a, b};
}













//##################################################################//
//                     33. Simple Range Queries                    //
class RangeQuery {
public:
	int *pref;
	RangeQuery(vi &a) {
		pref = new int[(int)a.size() + 5];
		pref[0] = 0;
		for0(a.sz) {
			pref[i + 1] = pref[i] + a[i];
		}
	}
	//1 based
	int query(int l, int r) {
		return pref[r] - pref[l - 1];
	}
};































//##################################################################//
//                     34. Least Prime Divisors                    //


vector<int> LeastPrimeDivisor(int n) {
	vector<int> least(n + 1);
	least[1] = 1;
	for (int i = 2; i <= n; i++) {
		if (least[i] == 0) {
			least[i] = i;
			for (int j = 2 * i; j <= n; j += i)
				if (least[j] == 0)
					least[j] = i;
		}
	}
	return least;
}












//##################################################################//
//                      35. Number to Binary                       //


string numtobin(int n) {
		string res = "";
		while (n > 0) {
			res.pb(n % 2 + '0');
			n /= 2;
		}
		while (res.ln < 10) {
			res.pb('0');
		}
		reverse(all(res));
	return res;
}











//##################################################################//
//              36. Strongly Connected Components                  //


class SCC {
public:
	map<int, set<int>> adj;
	map<int, set<int>> r_adj;
	vector<int> order;
	map<int, int> comp;
	int n;
	int m;

	SCC(int n, int m) {
		this->n = n;
		this->m = m;
	}

	void add_edge(int x, int y) {
		adj[x].insert(y);
		r_adj[y].insert(x);
	}

	void dfs(int src, map<int, int> &visit) {
		visit[src] = true;
		for (int x : adj[src]) {
			if (!visit[x]) {
				dfs(x, visit);
			}
		}
		order.push_back(src);
	}
	void dfs_rev(int src, map<int, int> &visit, int col) {

		visit[src] = true;
		comp[src] = col;
		for (int x : r_adj[src]) {
			if (!visit[x]) {
				dfs_rev(x, visit, col);
			}
		}
		order.push_back(src);
	}
	void create_order() {
		map<int, int> visit;
		for (int i = 1; i <= n; i++) {
			if (!visit[i]) {
				dfs(i, visit);
			}
		}
	}
	void Kosaraju() {
		map<int, int> visit;
		int col = 1;
		for (int i = n - 1; i >= 0; i--) {
			if (!visit[order[i]]) {
				dfs_rev(order[i], visit, col);
				col++;
			}
		}
	}

	map<int, int> get_SCC() {
		create_order();
		Kosaraju();
		return comp;
	}
};







//##################################################################//
//                   37. Offset Data Structure                      //

class Offset {
public:
	set<int> st;
	int delta;

	int fetch() {
		return (*st.begin()) + delta;
	}

	void increase(int x) {
		delta += x;
	}

	void add(int x) {
		st.insert(x - delta);
	}
};









//##################################################################//
//                   38. No Overflow Operations                 //


long long mulSafe(long long a, long long b) {
	long long temp = LLONG_MAX / a;
	if (temp <= (long long)b)return -1;
	return a * b;
}



long long addSafe(long long a, long long b) {
	long long temp = LLONG_MAX - a;
	if (temp <= (long long)b)return -1;
	return a + b;
}









//##################################################################//
//                      39.  Binary to Number                       //


int bintonum(string &n) {
	int res = 0;
	for (int i = n.length() - 1; i >= 0; i--) {
		if (n[i] == '0')continue;
		res += powl(2, n.length() - 1 - i);
	}
	return res;
}


















//##################################################################//
//          40.  Ternary Search For __/\___ like function           //

while (r - l > 2) {

	int m1 = l + (r - l) / 3;
	int m2 = r - (r - l) / 3;

	double f1 = f(m1);
	double f2 = f(m2);

	res = max(res, f1);
	res = max(res, f2);

	if (f1 < f2)
		l = m1;
	else
		r = m2;
}

for (int _i = l; _i <= r; _i++) {
	double cur = solve(a, k, _i, m);
	res = max(res, cur);
}
















//##################################################################//
//          40.  Ternary Search For __/\___ like function           //

while (r - l > 2) {

	int m1 = l + (r - l) / 3;
	int m2 = r - (r - l) / 3;

	double f1 = f(m1);
	double f2 = f(m2);

	res = max(res, f1);
	res = max(res, f2);

	if (f1 < f2)
		l = m1;
	else
		r = m2;
}

for (int _i = l; _i <= r; _i++) {
	double cur = solve(a, k, _i, m);
	res = max(res, cur);
}
























//##################################################################//
//                   41.  Find Centroids of a Tree                  //

vector<int> Centroid(map<int, set<int>> &adj, int n) {

	vector<int> centroid;
	map<int, int> siz;
	function<void (int, int)> dfs = [&](int u, int prev) {
		siz[u] = 1;
		bool is_centroid = true;
		for (auto v : adj[u])
			if (v != prev) {
				dfs(v, u);
				siz[u] += siz[v];
				if (siz[v] > n / 2) is_centroid = false;
			}
		if (n - siz[u] > n / 2) is_centroid = false;
		if (is_centroid) centroid.push_back(u);
	};
	dfs(1, -1);
	return centroid;
}


































//##################################################################//
//                      42.  Manacher Algorithm                    //

pair<int, int> Manacher(string a) {
	function<string (string &)> hashString = [](string & a) {
		string res = "";
		for (int i = 0; i < a.length(); i++) {
			res.push_back('#');
			res.push_back(a[i]);
		}
		res.push_back('#');
		return res;
	};
	function<pair<int, int> (string &a, pair<int, int> range)> get_range = [](string & a, pair<int, int> range) {
		int l = 0, r = 0;
		for (int i = 0; i <= range.first; i++) {
			if (a[i] != '#') {
				l++;
			}
		}
		r = l;
		for (int i = range.first; i <= range.second; i++) {
			if (a[i] != '#') {
				r++;
			}
		}
		pair<int, int> res = {l, r - 1};
		return res;
	};
	a = hashString(a);
	int l = 0, r = -1;
	vector<int> p(a.length());
	for (int i = 0; i < a.length(); i++) {
		int k;
		if (i > r) {
			k = 0;
		} else {
			int j = l + r - i;
			k = min(p[j], r - i);
		}

		while (i - k >= 0 and i + k < a.length() and a[i - k] == a[i + k]) {
			k++;
		}

		if (i - k >= 0 and i + k < a.length())
			k -= (a[i - k] != a[i + k]);
		else
			k--;

		p[i] = k;

		if (i + k > r) {
			l = i - k;
			r = i + k;
		}
	}

	int maxi = *max_element(p.begin(), p.end());
	pair<int, int> range_res;
	for (int i = 0; i < p.size(); i++) {
		if (p[i] == maxi) {
			range_res.first = max(0LL, i - maxi);
			range_res.second = min((int)a.length() - 1, i + maxi);
			break;
		}
	}

	pair<int, int> res = get_range(a, range_res);

	return res;
}






























//##################################################################//
//                        43.  Euler Toitient                        //
int totient(int n) {
	int result = n;
	for (int i = 2; i * i <= n; i++) {
		if (n % i == 0) {
			while (n % i == 0)
				n /= i;
			result -= result / i;
		}
	}
	if (n > 1)
		result -= result / n;
	return result;
}
















































//##################################################################//
//               44.  Big Interger Multiplication                   //


using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> & a, bool invert) {
	int n = a.size();
	if (n == 1)
		return;

	vector<cd> a0(n / 2), a1(n / 2);
	for (int i = 0; 2 * i < n; i++) {
		a0[i] = a[2 * i];
		a1[i] = a[2 * i + 1];
	}
	fft(a0, invert);
	fft(a1, invert);

	double ang = 2 * PI / n * (invert ? -1 : 1);
	cd w(1), wn(cos(ang), sin(ang));
	for (int i = 0; 2 * i < n; i++) {
		a[i] = a0[i] + w * a1[i];
		a[i + n / 2] = a0[i] - w * a1[i];
		if (invert) {
			a[i] /= 2;
			a[i + n / 2] /= 2;
		}
		w *= wn;
	}
}

vector<int> multiply(vector<int> & a, vector<int> & b) {
	vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
	int n = 1;
	while (n < a.size() + b.size())
		n <<= 1;
	fa.resize(n);
	fb.resize(n);

	fft(fa, false);
	fft(fb, false);
	for (int i = 0; i < n; i++)
		fa[i] *= fb[i];
	fft(fa, true);

	vector<int> result(n);
	for (int i = 0; i < n; i++)
		result[i] = round(fa[i].real());


	int carry = 0;
	for (int i = 0; i < n; i++) {
		result[i] += carry;
		carry = result[i] / 10;
		result[i] %= 10;
	}

	while (result.back() == 0) {
		result.pop_back();
	}
	reverse(result.begin(),result.begin());
	return result;
}

string bigmul(string &a, string &b) {
	vector<int> one ;
	while (a.length() > 0) {
		one.push_back(a.back() - '0');
		a.pop_back();
	}
	vector<int> two;
	while (b.length() > 0) {
		two.push_back(b.back() - '0');
		b.pop_back();
	}
	vector<int> ans = multiply(one, two);

	string res = "";
	for (int i = 0; i < ans.size(); i++) {
		res.push_back(ans[i] + '0');
	}
	return res;
}




























//##################################################################//
//                       45.  Big Interger C++                     //



class BigInteger {
	using cd = complex < double > ;
	const double PI = acos(-1);
	void fft(vector < cd > & a, bool invert) {
		int n = a.size();
		if (n == 1)
			return;
		vector < cd > a0(n / 2), a1(n / 2);
		for (int i = 0; 2 * i < n; i++) {
			a0[i] = a[2 * i];
			a1[i] = a[2 * i + 1];
		}
		fft(a0, invert);
		fft(a1, invert);
		double ang = 2 * PI / n * (invert ? -1 : 1);
		cd w(1), wn(cos(ang), sin(ang));
		for (int i = 0; 2 * i < n; i++) {
			a[i] = a0[i] + w * a1[i];
			a[i + n / 2] = a0[i] - w * a1[i];
			if (invert) {
				a[i] /= 2;
				a[i + n / 2] /= 2;
			}
			w *= wn;
		}
	}
	vector < int > multiply(vector < int > & a, vector < int > & b) {
		vector < cd > fa(a.begin(), a.end()), fb(b.begin(), b.end());
		int n = 1;
		while (n < a.size() + b.size())
			n <<= 1;
		fa.resize(n);
		fb.resize(n);
		fft(fa, false);
		fft(fb, false);
		for (int i = 0; i < n; i++)
			fa[i] *= fb[i];
		fft(fa, true);
		vector < int > rres(n);
		for (int i = 0; i < n; i++)
			rres[i] = round(fa[i].real());

		int carry = 0;
		for (int i = 0; i < n; i++) {
			rres[i] += carry;
			carry = rres[i] / 10;
			rres[i] %= 10;
		}
		while (rres.back() == 0) {
			rres.pop_back();
		}
		reverse(rres.begin(), rres.end());
		return rres;
	}
	string bigmul(string & a, string & b) {
		vector < int > one;
		while (a.length() > 0) {
			one.push_back(a.back() - '0');
			a.pop_back();
		}
		vector < int > two;
		while (b.length() > 0) {
			two.push_back(b.back() - '0');
			b.pop_back();
		}
		vector < int > ans = multiply(one, two);
		string res = "";
		for (int i = 0; i < ans.size(); i++) {
			res.push_back(ans[i] + '0');
		}
		return res;
	}

	string findSum(string str1, string str2) {
		if (str1.length() > str2.length()) swap(str1, str2);

		string str = "";

		int n1 = str1.length(), n2 = str2.length();

		reverse(str1.begin(), str1.end());
		reverse(str2.begin(), str2.end());

		int carry = 0;
		for (int i = 0; i < n1; i++) {
			int sum = ((str1[i] - '0') + (str2[i] - '0') + carry);
			str.push_back(sum % 10 + '0');
			carry = sum / 10;
		}

		for (int i = n1; i < n2; i++) {
			int sum = ((str2[i] - '0') + carry);
			str.push_back(sum % 10 + '0');
			carry = sum / 10;
		}
		if (carry) str.push_back(carry + '0');
		reverse(str.begin(), str.end());
		return str;
	}

	string num = "";
public:
	BigInteger(string a) {
		this -> num = a;
	}
	BigInteger() {

	}
	BigInteger operator * (BigInteger b) {
		string temp2 = b.num;
		string temp1 = this -> num;
		BigInteger res(bigmul(temp2, temp1));
		return res;
	}
	BigInteger operator + (BigInteger b) {
		string temp2 = b.num;
		string temp1 = this -> num;
		BigInteger res(findSum(temp2, temp1));
		return res;
	}
	friend ostream & operator << (ostream & out,
	                              const BigInteger & c) {
		out << c.num;
		return out;
	}
	friend istream & operator >> (istream & in , BigInteger & c) {
		in >> c.num;
		return in;
	}
};











//##################################################################//
//               46.  Count Subarray in an Array                    //


template < typename T >
int Count_Subarray_In_A_Array(vector<T> a,vector<T> b){

	function<vector<T> (vector<T>)> KMP_Preprocess = [&](vector<T> s) {
		vector<T> pi(s.size());
		for (int i = 1; i < s.size(); i++) {
			int j = pi[i - 1];
			while (j > 0 && s[i] != s[j])
				j = pi[j - 1];
			if (s[i] == s[j])
				j++;
			pi[i] = j;
		}
		return pi;
	};

	vector<T> lps=KMP_Preprocess(b);
	int i=0,j=0;
	int res=0;
	while(i<a.sz){
		if(j<b.sz and a[i]==b[j]){
			if(j==((int)b.sz - 1))res++;
			i++;
			j++;
		}else{
			if(j-1>=0){
				if(j==lps[j-1]){
					i++;
				}
				j=lps[j-1];
			}else{
				i++;
			}
		}
	}
	return res;
}








//##################################################################//
//                 47.  Get Diameter of a tree                      //


pair<int,int> getDiameter(int src,map<int,set<int>> &adj,int n) {

    int visit[200005],dist[200005],inQue[200005];
    memset(visit,0,sizeof visit);
    memset(dist,0,sizeof dist);
    memset(inQue,0,sizeof inQue);
    queue<int> Q;
    Q.push(src);
    dist[src]=0;
    while(!Q.empty()) {
        int cur = Q.front();
        Q.pop();
        visit[cur]=true;
        for(int x:adj[cur]) {
            if(!visit[x]) {
                dist[x]=dist[cur]+1;
                if(!inQue[x]) {
                    Q.push(x);
                    inQue[x]=true;
                }
            }
        }
    }
    int res=0;
    for(auto x:dist) {
        res=max(res,x);
    }
    for(int i=1; i<=n; i++) {
        if(res==dist[i]) {
            return {res,i};
        }
    }
}




























//##################################################################//
//                            48. DSU                              //

class UFDS{
	int n;
	vi par,siz;
	int comp;
	public:
	UFDS(int nval){
		n=nval;
		comp=n;
		par.resize(n+1);
		siz.resize(n+1);
		for(int i=0;i<n;i++){
			par[i+1]=i+1;
		}
	}
	int root(int node){
		if(node==par[node])return node;
		par[node]=root(par[node]);
		return par[node];
	}
	void uni(int a,int b){
		int r1=root(a),r2=root(b);
		if(r1==r2)return;
		// R1 is the new root
		par[r2]=r1;
	}
	bool find(int a,int b){
		return root(a)==root(b);
	}
};











//##################################################################//
//                            48. Split and Join                         //

template <typename T>
vector<vector<T>> split(vector<T> &a){
	function<bool (T, T)> cond = [&](T a, T b) {
		// Write Your Split Condition
		return false;
	};
	vector<vector<T>> res;
	vector<T> cur;
	cur.push_back(a[0]);
	for(int i=1;i<a.size();i++){
		if(cond(a[i-1],a[i])){
			res.push_back(cur);
			cur.clear();
		}
		cur.push_back(a[i]);
	}
	if(cur.size()>0)res.push_back(cur);
	return res;
}

template <typename T>
vector<T> join(vector<vector<T>> &a){
	vector<T> res;
	for(auto x:a) for(auto y:x) res.push_back(y);
	return res;
}













//##################################################################//
//                            48. Matrix Queries                    //


class MatrixQuery{
    vector<vector<int>> aux;
public:
    MatrixQuery(vector<vector<int> > &grid,int n,int m) {
        vector<vector<int> > temp(n,vector<int>(m));

        for (int i=0; i<m; i++)
            temp[0][i] = grid[0][i];

        for (int i=1; i<n; i++)
            for (int j=0; j<m; j++)
                temp[i][j] = grid[i][j] + temp[i-1][j];

        for (int i=0; i<n; i++)
            for (int j=1; j<m; j++)
                temp[i][j] += temp[i][j-1];
        aux=temp;
    }
    // (tli,tlj) - top-left
    // (rbi,rbj) - bottom-right
    int sumQuery(int tli, int tlj, int rbi,int rbj) {
        int res = aux[rbi][rbj];
        if (tli > 0)
            res = res - aux[tli-1][rbj];
        if (tlj > 0)
            res = res - aux[rbi][tlj-1];

        if (tli > 0 && tlj > 0)
            res = res + aux[tli-1][tlj-1];
        return res;
    }
};































//##################################################################//
//                            49. Bitmask                           //


class Bitmask{
	int mask=0;
public:
	Bitmask(int n){
		this->mask = powl(2,n)-1;	
	}
	// 0 based
	void setXBit(int idx){
		this->mask = this->mask|(1<<idx);
	}
	int getXBit(int idx){
		return (this->mask&(1<<idx))>0;
	}
	void clearXBit(int idx){
		this->setXBit(idx);
		this->mask = (this->mask)-(1<<idx);
	}
	int getMask(){
		return this->mask;
	}
	string show(){
		string res = "";
		int n = this->mask;
		while (n > 0) {
			res.pb(n % 2 + '0');
			n /= 2;
		}
		while (res.ln < 20) {
			res.pb('0');
		}

		return res;
	}
};






//##################################################################//
//                            50. SparseTable                           //



class SparseTable {
	vector<int> LOG;
	vector<vector<int>> tab;
	int K;
    private:
    int combine(int a,int b){
    	return a+b;
    }
    void buildSparseTable(vector<int> &a) {
		int n=a.size();
		for (int i = 0; i < n; i++)
			tab[i][0] = a[i];
		for (int j = 1; j < K; j++)
			for (int i = 0; i + (1 << j) <= n; i++)
				tab[i][j] = combine(tab[i][j-1], tab[i + (1 << (j - 1))][j - 1]);
    }
    
    public:    
    SparseTable(vector<int> &a) {
    	int n=a.size(),k=log2l(n)+1;
		this->K=k;
    	LOG.resize(n+1);
    	tab.resize(n+1,vector<int>(k));
        for (int i = 0; i <LOG.size(); i++) {
            LOG[i] = log2l(i);
        }
        buildSparseTable(a);
    }
    int query(int L, int R) {
        L--;
        R--;
        int j = LOG[R - L + 1];
        int res = combine(tab[L][j], tab[R - (1 << j) + 1][j]);
        return res;
    }
};