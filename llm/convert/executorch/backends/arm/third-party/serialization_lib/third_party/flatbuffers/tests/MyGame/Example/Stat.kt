// automatically generated by the FlatBuffers compiler, do not modify

package MyGame.Example

import com.google.flatbuffers.BaseVector
import com.google.flatbuffers.BooleanVector
import com.google.flatbuffers.ByteVector
import com.google.flatbuffers.Constants
import com.google.flatbuffers.DoubleVector
import com.google.flatbuffers.FlatBufferBuilder
import com.google.flatbuffers.FloatVector
import com.google.flatbuffers.LongVector
import com.google.flatbuffers.StringVector
import com.google.flatbuffers.Struct
import com.google.flatbuffers.Table
import com.google.flatbuffers.UnionVector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sign

@Suppress("unused")
class Stat : Table() {

    fun __init(_i: Int, _bb: ByteBuffer)  {
        __reset(_i, _bb)
    }
    fun __assign(_i: Int, _bb: ByteBuffer) : Stat {
        __init(_i, _bb)
        return this
    }
    val id : String?
        get() {
            val o = __offset(4)
            return if (o != 0) {
                __string(o + bb_pos)
            } else {
                null
            }
        }
    val idAsByteBuffer : ByteBuffer get() = __vector_as_bytebuffer(4, 1)
    fun idInByteBuffer(_bb: ByteBuffer) : ByteBuffer = __vector_in_bytebuffer(_bb, 4, 1)
    val val_ : Long
        get() {
            val o = __offset(6)
            return if(o != 0) bb.getLong(o + bb_pos) else 0L
        }
    fun mutateVal_(val_: Long) : Boolean {
        val o = __offset(6)
        return if (o != 0) {
            bb.putLong(o + bb_pos, val_)
            true
        } else {
            false
        }
    }
    val count : UShort
        get() {
            val o = __offset(8)
            return if(o != 0) bb.getShort(o + bb_pos).toUShort() else 0u
        }
    fun mutateCount(count: UShort) : Boolean {
        val o = __offset(8)
        return if (o != 0) {
            bb.putShort(o + bb_pos, count.toShort())
            true
        } else {
            false
        }
    }
    override fun keysCompare(o1: Int, o2: Int, _bb: ByteBuffer) : Int {
        val val_1 = _bb.getShort(__offset(8, o1, _bb))
        val val_2 = _bb.getShort(__offset(8, o2, _bb))
        return (val_1 - val_2).sign
    }
    companion object {
        fun validateVersion() = Constants.FLATBUFFERS_23_5_26()
        fun getRootAsStat(_bb: ByteBuffer): Stat = getRootAsStat(_bb, Stat())
        fun getRootAsStat(_bb: ByteBuffer, obj: Stat): Stat {
            _bb.order(ByteOrder.LITTLE_ENDIAN)
            return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb))
        }
        fun createStat(builder: FlatBufferBuilder, idOffset: Int, val_: Long, count: UShort) : Int {
            builder.startTable(3)
            addVal_(builder, val_)
            addId(builder, idOffset)
            addCount(builder, count)
            return endStat(builder)
        }
        fun startStat(builder: FlatBufferBuilder) = builder.startTable(3)
        fun addId(builder: FlatBufferBuilder, id: Int) = builder.addOffset(0, id, 0)
        fun addVal_(builder: FlatBufferBuilder, val_: Long) = builder.addLong(1, val_, 0L)
        fun addCount(builder: FlatBufferBuilder, count: UShort)  {
            builder.addShort(count.toShort())
            builder.slot(2)
        }
        fun endStat(builder: FlatBufferBuilder) : Int {
            val o = builder.endTable()
            return o
        }
        fun __lookup_by_key(obj: Stat?, vectorLocation: Int, key: UShort, bb: ByteBuffer) : Stat? {
            var span = bb.getInt(vectorLocation - 4)
            var start = 0
            while (span != 0) {
                var middle = span / 2
                val tableOffset = __indirect(vectorLocation + 4 * (start + middle), bb)
                val value = bb.getShort(__offset(8, bb.capacity() - tableOffset, bb)).toUShort()
                val comp = value.compareTo(key)
                when {
                    comp > 0 -> span = middle
                    comp < 0 -> {
                        middle++
                        start += middle
                        span -= middle
                    }
                    else -> {
                        return (obj ?: Stat()).__assign(tableOffset, bb)
                    }
                }
            }
            return null
        }
    }
}