#ifndef NBR_H_
#define NBR_H_

#include "nbr/abstract_nbr.h"
#include "nbr/dummy_nbr.h"
#include "nbr/pq_nbr.h"
#include "nbr/rabitq_nbr.h"

namespace pipeann {
  /* Neighbor handler is used to compute distances between query and graph neighbors. */
  template<typename T>
  inline AbstractNeighbor<T> *get_nbr_handler(Metric metric, const std::string &nbr_type) {
    if (nbr_type == "rabitq") {
      return new RaBitQNeighbor<T>(metric);
    } else if (nbr_type == "pq") {
      return new PQNeighbor<T>(metric);
    } else if (nbr_type == "dummy") {
      return new DummyNeighbor<T>(metric);
    } else if (nbr_type == "rabitq3") {
      return new RaBitQNeighbor<T, 3>(metric);
    } else if (nbr_type == "rabitq4") {
      return new RaBitQNeighbor<T, 4>(metric);
    } else if (nbr_type == "rabitq5") {
      return new RaBitQNeighbor<T, 5>(metric);
    }
    return nullptr;
  }
}  // namespace pipeann

#endif  // NBR_H_