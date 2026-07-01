#include <iostream>
#include <thread>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <queue>
#include <cassert>
#include <fstream>
#include <condition_variable>
#include <atomic>
#include <vector>

using std::cout;
using std::endl;

std::ifstream in;
std::ofstream out;
int T, L_min, L_max;
std::size_t M;

std::queue<std::string> buffer;

std::mutex read_mtx, write_mtx, buffer_mtx;
std::mutex producer_push_mutex;
std::condition_variable empty_cv; // notify producers when space available
std::condition_variable fill_cv;  // notify consumers when data available

std::atomic<int> producers_alive{0};

inline int getRandomInt()
{
    thread_local std::mt19937 gen(std::random_device{}()); // mt19937 is not thread-safe
    std::uniform_int_distribution<int> dist(L_min, L_max);
    return dist(gen);
}

// RAII guard for producer thread, so that it decrements the counter on scope exit
struct ProducerGuard
{
    std::atomic<int> &counter;
    ProducerGuard(std::atomic<int> &c) : counter(c) {}
    ~ProducerGuard() { counter.fetch_sub(1); }
};

void producer()
{
    ProducerGuard guard(producers_alive); // ensures decrement at scope exit
    while (true)
    {
        std::vector<std::string> dataRead; // buffer for data read from file
        int L = getRandomInt();            // random number of lines to read
        dataRead.reserve(L);               // reserve space for lines
        int i = 0;
        {
            std::unique_lock<std::mutex> rlock(read_mtx); // The Input file is a shared Resource, hence we need synchronization to prevent data races
            std::string line;
            for (; i < L && std::getline(in, line); ++i)
            { // Reads L lines from the file (or until EOF)
                dataRead.push_back(line);
            }
        }
        if (i == 0)
        { // If no data Read, EOF has been received
            break;
        }
        {
            std::unique_lock<std::mutex> prod_write_lock(producer_push_mutex); // To Atomically write the L lines to the buffer
            size_t idx = 0;
            while (idx < dataRead.size())
            {
                std::unique_lock<std::mutex> bl(buffer_mtx); // Buffer is a Shared Resource as well, need a lock for access
                empty_cv.wait(bl, [&]
                              { return buffer.size() < M; }); // Wait until there is space in the buffer

                while (idx < dataRead.size() && buffer.size() < M)
                {                                            // push as many as fit
                    buffer.push(std::move(dataRead[idx++])); // Move data into the buffer
                }
                fill_cv.notify_one(); // notify a consumer that data is available
            }
        }
        if (i < L)
        { // If fewer lines were read than requested, we reached EOF
            break;
        }
    }
    fill_cv.notify_all(); // notify all consumers so that any waiting consumers can wake up
}

void consumer()
{
    std::vector<std::string> dataWrite; // buffer for data to write
    while (true)
    {
        {
            std::lock_guard<std::mutex> wlock(write_mtx); // Guard for writing to output file as it is also a shared Resource
            std::unique_lock<std::mutex> bl(buffer_mtx);  // Lock for accessing the buffer
            fill_cv.wait(bl, [&]
                         { return !buffer.empty() || producers_alive.load(std::memory_order_acquire) == 0; });

            if (buffer.empty() && producers_alive.load(std::memory_order_acquire) == 0)
            {
                // No more incoming producers and nothing in buffer => we can exit, lock guard handles the unlocking of the wlock
                break;
            }
            while (!buffer.empty())
            {
                dataWrite.push_back(buffer.front()); // Move data from buffer to local vector
                buffer.pop();
            }
            empty_cv.notify_one(); // notify a producer that there is space in the buffer
            for (auto &s : dataWrite)
            { // Write the data to the output file
                out << s << "\n";
            }
            dataWrite.clear();
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        throw std::runtime_error("Format: ./" + std::string(argv[0]) + " <input_file> <threads> <L_min> <L_max> <buffer_size> <output_file>");
    }

    in.open(argv[1]);
    if (!in)
        throw std::runtime_error("Failed to open input file " + std::string(argv[1]));

    T = std::stoi(argv[2]);
    L_min = std::stoi(argv[3]);
    L_max = std::stoi(argv[4]);
    M = static_cast<std::size_t>(std::stoul(argv[5]));

    if (T <= 0)         throw std::invalid_argument("Number of threads must be positive");
    if (L_min <= 0)     throw std::invalid_argument("Minimum line count must be positive");
    if (L_max < L_min)  throw std::invalid_argument("Maximum line count must be at least minimum");
    if (M == 0)         throw std::invalid_argument("Buffer size must be positive");

    out.open(argv[6], std::ios::out); // Open output file
    if (!out)
        throw std::runtime_error("Failed to open output file " + std::string(argv[6]));

    producers_alive.store(T, std::memory_order_release); // Set the atomic counter for active producers

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    for (int i = 0; i < T; ++i)
        producers.emplace_back(producer);
    consumers.emplace_back(consumer); // at least one consumer is needed
    for (int i = 1; i < T / 2; ++i)
        consumers.emplace_back(consumer);

    for (auto &t : producers)
        t.join();         // Wait for all producers to finish
    fill_cv.notify_all(); // notify all consumers so that any waiting consumers can wake up
    for (auto &t : consumers)
        t.join(); // Wait for all consumers to finish
    in.close();   // Close input file
    out.close();  // Close output file
    return 0;
}
